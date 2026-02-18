"""
threej_cosmo: Fast Wigner 3j symbol-based coupling matrices for CMB analysis.

This package provides efficient computation of mode-coupling matrices K^TT and K^EE
using OpenMP-parallelized C code with Python bindings. GPU acceleration is available
on systems with NVIDIA or AMD GPUs and appropriate compilers.

Example usage:
    import numpy as np
    from threej_cosmo import coupling_matrix_TT, coupling_matrix_EE

    # Load your window function Cls
    window_cls = np.fromfile("window_cls.bin", dtype=np.float64)

    # Compute coupling matrices (CPU)
    K_TT = coupling_matrix_TT(window_cls, lmax=2000)
    K_EE = coupling_matrix_EE(window_cls, lmax=2000)

    # Use GPU acceleration if available
    K_TT = coupling_matrix_TT(window_cls, lmax=2000, backend="gpu")

    # Check available backends
    from threej_cosmo import get_available_backends
    print(get_available_backends())  # ['cpu'] or ['cpu', 'gpu']
"""

from .coupling import (
    coupling_matrix_TT,
    coupling_matrix_EE,
    coupling_matrix_EB,
)
from .utils import (
    ensure_binaries_compiled,
    get_binary_path,
    get_available_backends,
)

__version__ = "0.1.0"
__author__ = "Georgia Kiddier"

__all__ = [
    "coupling_matrix_TT",
    "coupling_matrix_EE",
    "coupling_matrix_EB",
    "ensure_binaries_compiled",
    "get_binary_path",
    "get_available_backends",
]
