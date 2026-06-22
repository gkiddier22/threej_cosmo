"""
Core coupling matrix computation functions.
"""

import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Literal

import numpy as np

from .utils import (
    get_binary_path,
    ensure_binaries_compiled,
    get_available_backends,
)


BackendType = Literal["auto", "cpu", "gpu"]


def _resolve_backend(backend: BackendType) -> str:
    """
    Resolve the backend to use.

    Parameters
    ----------
    backend : str
        'auto', 'cpu', or 'gpu'

    Returns
    -------
    str
        'cpu' or 'gpu'
    """
    if backend == "cpu":
        return "cpu"
    elif backend == "gpu":
        return "gpu"
    elif backend == "auto":
        available = get_available_backends()
        if "gpu" in available:
            # Check if GPU binaries exist, if so prefer GPU
            from .utils import get_src_gpu_dir
            gpu_dir = get_src_gpu_dir()
            if (gpu_dir / "thrj_000_gpu").exists():
                return "gpu"
        if "cpu" in available:
            return "cpu"
        raise RuntimeError("No backends available. Install gcc for CPU support.")
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'cpu', or 'gpu'.")


def _run_coupling_program(
    binary_name: str,
    window_cls: np.ndarray,
    lmax: int,
    extra_args: Optional[list] = None,
    verbose: bool = False,
    backend: BackendType = "cpu",
) -> np.ndarray:
    """
    Run a coupling matrix C program and return the result.

    Parameters
    ----------
    binary_name : str
        Name of the binary ('thrj_all')
    window_cls : np.ndarray
        Window function Cls, must have at least 2*lmax+1 elements
    lmax : int
        Maximum multipole for the coupling matrix
    extra_args : list, optional
        Additional command-line arguments
    verbose : bool
        Print timing information
    backend : str
        'auto', 'cpu', or 'gpu'

    Returns
    -------
    np.ndarray
        Coupling matrix of shape (lmax+1, lmax+1)
    """
    # Resolve backend
    resolved_backend = _resolve_backend(backend)

    # Ensure binaries are compiled
    try:
        ensure_binaries_compiled(backend=resolved_backend)
    except RuntimeError as e:
        if resolved_backend == "gpu" and backend == "auto":
            # Fallback to CPU if GPU compilation fails in auto mode
            warnings.warn(f"GPU compilation failed, falling back to CPU: {e}")
            resolved_backend = "cpu"
            ensure_binaries_compiled(backend="cpu")
        else:
            raise

    binary_path = get_binary_path(binary_name, backend=resolved_backend)
    if not binary_path.exists():
        raise FileNotFoundError(
            f"Binary not found: {binary_path}. "
            f"Run ensure_binaries_compiled(backend='{resolved_backend}') first."
        )

    # Validate input
    window_cls = np.asarray(window_cls, dtype=np.float64)
    required_size = 2 * lmax + 1
    if len(window_cls) < required_size:
        raise ValueError(
            f"window_cls has {len(window_cls)} elements, need at least {required_size} for lmax={lmax}"
        )

    # Truncate to required size
    window_cls = window_cls[:required_size].copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        window_file = tmpdir / "window.bin"
        output_file = tmpdir / "coupling.bin"

        # Write window function
        window_cls.tofile(window_file)

        # Build command
        cmd = [str(binary_path), str(window_file), str(output_file), str(lmax)]
        if extra_args:
            cmd.extend(extra_args)
        cmd.append("yes")  # Always write output

        # Run
        result = subprocess.run(cmd, capture_output=True, text=True)

        if verbose:
            backend_str = f"[{resolved_backend.upper()}]"
            if result.stdout:
                print(f"{backend_str} {result.stdout.strip()}")

        if result.returncode != 0:
            error_msg = f"Coupling computation failed ({resolved_backend}):\n{result.stderr}"
            if resolved_backend == "gpu" and backend == "auto":
                # Try falling back to CPU
                warnings.warn(f"GPU execution failed, falling back to CPU: {result.stderr}")
                return _run_coupling_program(
                    binary_name, window_cls, lmax, extra_args, verbose, backend="cpu"
                )
            raise RuntimeError(error_msg)

        # Read result
        matrix = np.fromfile(output_file, dtype=np.float64)
        matrix = matrix.reshape((lmax + 1, lmax + 1))

    return matrix


def coupling_matrix_TT(
    window_cls: np.ndarray,
    lmax: int,
    verbose: bool = False,
    backend: BackendType = "cpu",
) -> np.ndarray:
    """
    Compute the TT (temperature) coupling matrix.

    Parameters
    ----------
    window_cls : np.ndarray
        Window function power spectrum Cls from l=0 to l=2*lmax.
        Can be computed from a mask using healpy.anafast().
    lmax : int
        Maximum multipole for the output coupling matrix.
        The matrix will have shape (lmax+1, lmax+1).
    verbose : bool, optional
        Print timing information. Default False.
    backend : str, optional
        Computation backend: 'auto', 'cpu', or 'gpu'. Default 'cpu'.
        - 'cpu': Use OpenMP-parallelized CPU code
        - 'gpu': Use GPU-accelerated code (requires NVIDIA/AMD GPU compiler)
        - 'auto': Use GPU if available, otherwise CPU

    Returns
    -------
    np.ndarray
        Coupling matrix K_TT of shape (lmax+1, lmax+1).
        Element K[l1, l2] gives the coupling between multipoles l1 and l2.

    Examples
    --------
    >>> import numpy as np
    >>> from threej_cosmo import coupling_matrix_TT
    >>> window_cls = np.fromfile("window_cls.bin", dtype=np.float64)
    >>> K_TT = coupling_matrix_TT(window_cls, lmax=2000)
    >>> print(K_TT.shape)
    (2001, 2001)

    >>> # Use GPU if available
    >>> K_TT = coupling_matrix_TT(window_cls, lmax=2000, backend="gpu")
    """
    return _run_coupling_program(
        "thrj_all", window_cls, lmax, extra_args=["TT"], verbose=verbose, backend=backend
    )


def coupling_matrix_EE(
    window_cls: np.ndarray,
    lmax: int,
    verbose: bool = False,
    backend: BackendType = "cpu",
) -> np.ndarray:
    """
    Compute the EE (E-mode polarization) coupling matrix.

    Parameters
    ----------
    window_cls : np.ndarray
        Window function power spectrum Cls from l=0 to l=2*lmax.
        Can be computed from a mask using healpy.anafast().
    lmax : int
        Maximum multipole for the output coupling matrix.
        The matrix will have shape (lmax+1, lmax+1).
    verbose : bool, optional
        Print timing information. Default False.
    backend : str, optional
        Computation backend: 'auto', 'cpu', or 'gpu'. Default 'cpu'.
        - 'cpu': Use OpenMP-parallelized CPU code
        - 'gpu': Use GPU-accelerated code (requires NVIDIA/AMD GPU compiler)
        - 'auto': Use GPU if available, otherwise CPU

    Returns
    -------
    np.ndarray
        Coupling matrix K_EE of shape (lmax+1, lmax+1).
        Element K[l1, l2] gives the coupling between multipoles l1 and l2.

    Examples
    --------
    >>> import numpy as np
    >>> from threej_cosmo import coupling_matrix_EE
    >>> window_cls = np.fromfile("window_cls.bin", dtype=np.float64)
    >>> K_EE = coupling_matrix_EE(window_cls, lmax=2000)
    >>> print(K_EE.shape)
    (2001, 2001)
    """
    return _run_coupling_program(
        "thrj_all", window_cls, lmax, extra_args=["EE"], verbose=verbose, backend=backend
    )


def coupling_matrix_TE(
    window_cls: np.ndarray,
    lmax: int,
    verbose: bool = False,
    backend: BackendType = "cpu",
) -> np.ndarray:
    """
    Compute the TE coupling matrix.

    Parameters
    ----------
    window_cls : np.ndarray
        Window function power spectrum Cls from l=0 to l=2*lmax.
        Can be computed from a mask using healpy.anafast().
    lmax : int
        Maximum multipole for the output coupling matrix.
        The matrix will have shape (lmax+1, lmax+1).
    verbose : bool, optional
        Print timing information. Default False.
    backend : str, optional
        Computation backend: 'auto', 'cpu', or 'gpu'. Default 'cpu'.
        - 'cpu': Use OpenMP-parallelized CPU code
        - 'gpu': Use GPU-accelerated code (requires NVIDIA/AMD GPU compiler)
        - 'auto': Use GPU if available, otherwise CPU

    Returns
    -------
    np.ndarray
        Coupling matrix K_EE of shape (lmax+1, lmax+1).
        Element K[l1, l2] gives the coupling between multipoles l1 and l2.

    Examples
    --------
    >>> import numpy as np
    >>> from threej_cosmo import coupling_matrix_EE
    >>> window_cls = np.fromfile("window_cls.bin", dtype=np.float64)
    >>> K_TE = coupling_matrix_TE(window_cls, lmax=2000)
    >>> print(K_TE.shape)
    (2001, 2001)
    """
    return _run_coupling_program(
        "thrj_all", window_cls, lmax, extra_args=["TE"], verbose=verbose, backend=backend
    )


def coupling_matrix_EB(
    window_cls: np.ndarray,
    lmax: int,
    verbose: bool = False,
    backend: BackendType = "cpu",
) -> np.ndarray:
    """
    Compute the EB (E-to-B polarization) coupling matrix.

    Parameters
    ----------
    window_cls : np.ndarray
        Window function power spectrum Cls from l=0 to l=2*lmax.
        Can be computed from a mask using healpy.anafast().
    lmax : int
        Maximum multipole for the output coupling matrix.
        The matrix will have shape (lmax+1, lmax+1).
    verbose : bool, optional
        Print timing information. Default False.
    backend : str, optional
        Computation backend: 'auto', 'cpu', or 'gpu'. Default 'cpu'.
        - 'cpu': Use OpenMP-parallelized CPU code
        - 'gpu': Use GPU-accelerated code (requires NVIDIA/AMD GPU compiler)
        - 'auto': Use GPU if available, otherwise CPU

    Returns
    -------
    np.ndarray
        Coupling matrix K_EB of shape (lmax+1, lmax+1).
        Element K[l1, l2] gives the coupling between multipoles l1 and l2.

    Examples
    --------
    >>> import numpy as np
    >>> from threej_cosmo import coupling_matrix_EB
    >>> window_cls = np.fromfile("window_cls.bin", dtype=np.float64)
    >>> K_EB = coupling_matrix_EB(window_cls, lmax=2000)
    >>> print(K_EB.shape)
    (2001, 2001)
    """
    return _run_coupling_program(
        "thrj_all", window_cls, lmax, extra_args=["EB"], verbose=verbose, backend=backend
    )
