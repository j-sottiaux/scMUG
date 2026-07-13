"""Training loop for scDMKC.

Two phases (the paper's lr = 1e-6 strongly implies a fine-tuning stage that
follows a higher-lr pretraining; the pretraining itself is not described).

# ASSUMPTION: phase 1 pretrains the encoder + kernel learner + decoder with the
# reconstruction objective (L_r + L_ZINB) at a higher learning rate; phase 2 runs
# the full joint objective (eq. 11) at the paper's settings. Both phases are
# configurable and pretraining can be disabled.

The trained estimator exposes three "exit points" used by the scMUG integration
(minimal-replacement option (a)):
    .latent_                 -> H^(L), the bottleneck embedding (N, d_latent)
    .kernel_representation_  -> K, the normalized cell-cell representation (N, N)
    .labels_                 -> final k-means labels on K
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.cluster import KMeans

from .losses import clustering_loss, kernel_loss, representation_loss
from .metrics import ari as _ari
from .metrics import nmi as _nmi
from .model import ScDMKC
from .zinb import ZINBLoss


class ScDMKCTrainer:
    def __init__(
        self,
        n_clusters: int,
        encoder_hidden=(500, 500, 2000, 10),
        decoder_hidden=(2000, 500, 500),
        kernels=("sigmoid", "cosine", "polynomial", "gaussian"),
        kernel_kwargs: dict | None = None,
        lambda1: float = 0.1,  # kernel loss weight
        lambda2: float = 1.0,  # clustering loss weight
        lambda3: float = 0.05,  # ZINB loss weight
        alpha: float = 1.0,  # Student-t dof
        pretrain_epochs: int = 300,
        pretrain_lr: float = 1e-3,
        n_iter: int = 200,
        lr: float = 1e-4,
        update_interval: int = 1,
        tol: float = 1e-3,
        min_iter: int = 20,
        kmeans_n_init: int = 20,
        normalize_kernel_features: bool = True,
        device: str | None = None,
        seed: int = 0,
        verbose: bool = True,
    ):
        self.n_clusters = n_clusters
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.kernels = kernels
        self.kernel_kwargs = kernel_kwargs
        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3
        self.alpha = alpha
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.n_iter = n_iter
        self.lr = lr
        self.update_interval = update_interval
        self.tol = tol
        self.min_iter = min_iter
        self.kmeans_n_init = kmeans_n_init
        self.normalize_kernel_features = normalize_kernel_features
        self.seed = seed
        self.verbose = verbose
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model: ScDMKC | None = None
        self.latent_ = None
        self.kernel_representation_ = None
        self.labels_ = None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _as_2d_float_array(name, value) -> np.ndarray:
        if hasattr(value, "toarray"):
            value = value.toarray()
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"{name} must be non-empty, got shape {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _as_size_factors(value, n_cells: int) -> np.ndarray:
        if hasattr(value, "toarray"):
            value = value.toarray()
        sf = np.asarray(value, dtype=np.float32)
        if sf.ndim == 1:
            sf = sf.reshape(-1, 1)
        if sf.shape != (n_cells, 1):
            raise ValueError(
                "size_factors must have shape (n_cells, 1), "
                f"got {sf.shape} for n_cells={n_cells}."
            )
        if not np.isfinite(sf).all():
            raise ValueError("size_factors contains NaN or infinite values.")
        if np.any(sf <= 0):
            raise ValueError("size_factors must be strictly positive.")
        return np.ascontiguousarray(sf, dtype=np.float32)

    def _validate_fit_inputs(self, X_input, X_raw, size_factors):
        X_arr = self._as_2d_float_array("X_input", X_input)
        Xr_arr = self._as_2d_float_array("X_raw", X_raw)
        if X_arr.shape != Xr_arr.shape:
            raise ValueError(
                "X_input and X_raw must have the same shape; "
                f"got {X_arr.shape} and {Xr_arr.shape}."
            )
        if np.any(Xr_arr < 0):
            raise ValueError("X_raw must be non-negative counts for the ZINB loss.")
        if not np.allclose(Xr_arr, np.rint(Xr_arr), atol=1e-3):
            warnings.warn(
                "X_raw is not integer-valued. ZINB is a count likelihood; "
                "treat this run as an approximation or pass raw counts.",
                RuntimeWarning,
                stacklevel=2,
            )
        n_cells = X_arr.shape[0]
        if self.n_clusters > n_cells:
            raise ValueError(
                f"n_clusters={self.n_clusters} cannot exceed n_cells={n_cells}."
            )
        if self.update_interval <= 0:
            raise ValueError("update_interval must be strictly positive.")
        if self.kmeans_n_init <= 0:
            raise ValueError("kmeans_n_init must be strictly positive.")
        if size_factors is None:
            sf_arr = np.ones((n_cells, 1), dtype=np.float32)
        else:
            sf_arr = self._as_size_factors(size_factors, n_cells)
        return X_arr, Xr_arr, sf_arr

    def _to_tensor(self, a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def _kmeans_labels(self, K: np.ndarray) -> np.ndarray:
        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.kmeans_n_init,
            random_state=self.seed,
        )
        return km.fit_predict(K)

    # -------------------------------------------------------------------- fit
    def fit(self, X_input, X_raw, size_factors=None, y=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_arr, Xr_arr, sf_arr = self._validate_fit_inputs(X_input, X_raw, size_factors)
        X = self._to_tensor(X_arr)  # (N, G) encoder input / recon target
        Xr = self._to_tensor(Xr_arr)  # (N, G) raw counts for ZINB
        sf = self._to_tensor(sf_arr)
        n_cells, n_genes = X.shape

        self.model = ScDMKC(
            n_cells=n_cells,
            n_genes=n_genes,
            n_clusters=self.n_clusters,
            encoder_hidden=self.encoder_hidden,
            decoder_hidden=self.decoder_hidden,
            kernels=self.kernels,
            kernel_kwargs=self.kernel_kwargs,
            alpha=self.alpha,
            normalize_kernel_features=self.normalize_kernel_features,
        ).to(self.device)
        zinb = ZINBLoss().to(self.device)

        # ---- phase 1: pretraining (reconstruction) ------------------------
        if self.pretrain_epochs > 0:
            opt = torch.optim.Adam(
                [
                    p
                    for n, p in self.model.named_parameters()
                    if not n.startswith("cluster_centers")
                ],
                lr=self.pretrain_lr,
            )
            for ep in range(self.pretrain_epochs):
                self.model.train()
                opt.zero_grad()
                out = self.model(X)
                loss = representation_loss(X, out["x_prime"]) + self.lambda3 * zinb(
                    Xr, out["mu"], out["theta"], out["pi"], sf
                )
                loss.backward()
                opt.step()
                if self.verbose and (ep % 50 == 0 or ep == self.pretrain_epochs - 1):
                    print(f"[pretrain {ep:4d}] loss={loss.item():.4f}")

        # ---- initialise cluster centers from k-means on K -----------------
        self.model.eval()
        with torch.no_grad():
            K0 = self.model(X)["K"].cpu().numpy()
        init_labels = self._kmeans_labels(K0)
        if np.unique(init_labels).size != self.n_clusters:
            raise RuntimeError(
                "Initial K-means did not return all requested clusters; "
                "check n_clusters, duplicated cells, or K degeneracy."
            )
        centers = np.stack(
            [K0[init_labels == c].mean(0) for c in range(self.n_clusters)]
        )
        self.model.cluster_centers.data = self._to_tensor(centers)
        prev_labels = init_labels

        # diagnostic: ground truth (eval only, never used for optimisation)
        y_eval = None
        if y is not None:
            _, y_eval = np.unique(np.asarray(y), return_inverse=True)
            if self.verbose:
                print(
                    f"[baseline] post-pretrain k-means  "
                    f"NMI={_nmi(y_eval, init_labels):.4f} "
                    f"ARI={_ari(y_eval, init_labels):.4f}"
                )

        # ---- phase 2: joint optimisation (eq. 11) -------------------------
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        p_target = None
        eval_count = 0
        for it in range(self.n_iter):
            self.model.train()

            if it % self.update_interval == 0:
                with torch.no_grad():
                    q = self.model(X)["q"]
                    p_target = ScDMKC.target_distribution(q).detach()
                hard = q.argmax(1).cpu().numpy()
                delta = np.mean(hard != prev_labels)
                prev_labels = hard
                if (
                    self.verbose
                    and y_eval is not None
                    and (eval_count % 10 == 0 or it >= self.n_iter - 1)
                ):
                    print(
                        f"[joint {it:4d}] q  NMI={_nmi(y_eval, hard):.4f} "
                        f"ARI={_ari(y_eval, hard):.4f}  (delta={delta:.4f})"
                    )
                eval_count += 1
                if it >= self.min_iter and delta < self.tol:
                    if self.verbose:
                        print(f"[joint {it}] label change {delta:.4f} < tol, stop")
                    break

            opt.zero_grad()
            out = self.model(X)
            assignments = out["q"].argmax(1)
            L_r = representation_loss(X, out["x_prime"])
            L_k = kernel_loss(
                out["hs"], out["K"], self.model.cluster_centers, assignments
            )
            L_c = clustering_loss(p_target, out["q"])
            L_z = zinb(Xr, out["mu"], out["theta"], out["pi"], sf)
            loss = L_r + self.lambda1 * L_k + self.lambda2 * L_c + self.lambda3 * L_z
            loss.backward()
            opt.step()

            if self.verbose and (it % 50 == 0 or it == self.n_iter - 1):
                print(
                    f"[joint {it:4d}] L={loss.item():.4f} "
                    f"(r={L_r.item():.3f} k={L_k.item():.3f} "
                    f"c={L_c.item():.4f} z={L_z.item():.3f})"
                )

        # ---- final outputs / exit points ----------------------------------
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)

            self.kernel_representation_ = out["K"].detach().cpu().numpy()
            self.latent_ = out["hs"][-1].detach().cpu().numpy()
            self.z_representation_ = self.latent_

            self.q_labels_ = out["q"].argmax(1).detach().cpu().numpy()

        # paper-style final clustering: K-means on K
        self.labels_ = self._kmeans_labels(self.kernel_representation_)

        return self

    # --------------------------------------------------------------- predict
    def predict(self):
        """Return the final k-means labels on K (clustering is transductive)."""
        if self.labels_ is None:
            raise RuntimeError("Call fit() first.")
        return self.labels_
