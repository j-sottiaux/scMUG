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
    .kernel_representation_  -> K, the consistent kernel matrix (N, N)
    .labels_                 -> final k-means labels on K
"""

from __future__ import annotations

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
        n_iter: int = 300,
        lr: float = 1e-4,
        update_interval: int = 1,
        tol: float = 1e-3,
        min_iter: int = 20,
        kmeans_n_init: int = 20,
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

        X = self._to_tensor(X_input)  # (N, G) encoder input / recon target
        Xr = self._to_tensor(X_raw)  # (N, G) raw counts for ZINB
        sf = (
            self._to_tensor(size_factors)
            if size_factors is not None
            else torch.ones(X.shape[0], 1, device=self.device)
        )
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
