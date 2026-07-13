"""dmkcn: a stabilized PyTorch implementation inspired by scDMKC.

Yao & Ren (2025), "Deep multi-kernel cell clustering for single-cell RNA
sequencing data", Biochemical Engineering Journal 223, 109877.

Public API
----------
    preprocess              scRNA-seq preprocessing pipeline (Section 4.1)
    ScDMKC                  the nn.Module (encoder + multi-kernel + ZINB decoder)
    ScDMKCTrainer           pretraining + joint training + fit/predict
    evaluate, nmi, ari, acc clustering metrics

The trainer exposes the exit points used by the scMUG integration:
    trainer.kernel_representation_ -> K, the (N, N) normalized cell-cell
                                      representation used for final K-means
                                      and spectral scMUG integration
    trainer.latent_                -> H^(L) bottleneck embedding for diagnostics
    trainer.labels_                -> final k-means cluster labels
"""

from .metrics import acc, ari, evaluate, nmi
from .model import ScDMKC
from .preprocessing import PreprocessedData, from_scmug_anndata, preprocess
from .trainer import ScDMKCTrainer

__all__ = [
    "preprocess", "PreprocessedData", "from_scmug_anndata",
    "ScDMKC", "ScDMKCTrainer",
    "evaluate", "nmi", "ari", "acc",
]

__version__ = "0.1.0"
