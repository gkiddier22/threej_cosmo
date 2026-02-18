# threej_cosmo

Fast computation of CMB mode-coupling matrices using optimised Wigner 3j symbol evaluation.

This package computes the coupling matrices $K^{TT}$, $K^{EE}$, and $K^{EB}$ required for pseudo-Cl power spectrum estimation on the masked sky. The method exploits analytic structure in the Wigner 3j symbols to achieve O(10) speedups over standard recursion-based approaches (e.g. pspy/MASTER), with additional O(10) acceleration on GPUs.

Based on Kiddier & Gratton (2026).

## Installation

```bash
pip install git+https://github.com/gkiddier22/threej_cosmo.git
```

Requires a C compiler with OpenMP support. On macOS: `brew install gcc`

## Quick Start

```python
import numpy as np
import healpy as hp
from threej_cosmo import coupling_matrix_TT, coupling_matrix_EE, coupling_matrix_EB

# 1. Compute window function Cls from your mask
mask = hp.read_map("mask.fits")
lmax = 2000
window_cls = hp.anafast(mask, lmax=2*lmax)

# 2. Compute coupling matrices
K_TT = coupling_matrix_TT(window_cls, lmax=lmax)
K_EE = coupling_matrix_EE(window_cls, lmax=lmax)
K_EB = coupling_matrix_EB(window_cls, lmax=lmax)
```

## API

### `coupling_matrix_TT(window_cls, lmax, verbose=False, backend="cpu")`

Compute the temperature coupling matrix using (0,0,0) Wigner 3j symbols.

### `coupling_matrix_EE(window_cls, lmax, verbose=False, backend="cpu")`

Compute the E-mode polarization coupling matrix using (-2,2,0) Wigner 3j symbols (even parity).

### `coupling_matrix_EB(window_cls, lmax, verbose=False, backend="cpu")`

Compute the E-to-B coupling matrix using (-2,2,0) Wigner 3j symbols (odd parity).

**Parameters:**
- `window_cls`: Window function power spectrum from l=0 to l=2*lmax (computed from mask via `healpy.anafast`)
- `lmax`: Maximum multipole for output matrix
- `verbose`: Print timing information
- `backend`: `"cpu"`, `"gpu"`, or `"auto"`

**Returns:** Coupling matrix K of shape `(lmax+1, lmax+1)`, where `K[l1, l2]` gives the coupling between multipoles l1 and l2.

## Pseudo-Cl Integration

```python
import numpy as np
import healpy as hp
from threej_cosmo import coupling_matrix_TT

# Compute coupling matrix
mask = hp.read_map("mask.fits")
lmax = 2000
window_cls = hp.anafast(mask, lmax=2*lmax)
K_TT = coupling_matrix_TT(window_cls, lmax=lmax)

# Compute pseudo-Cls from masked map
masked_map = hp.read_map("map.fits") * mask
pseudo_cls = hp.anafast(masked_map, lmax=lmax)

# Deconvolve to recover true Cls (typically done with binning)
K_inv = np.linalg.inv(K_TT[2:, 2:])
deconvolved_cls = K_inv @ pseudo_cls[2:lmax+1]
```

## GPU Support

GPU acceleration is available on systems with NVIDIA GPUs and the NVIDIA HPC SDK:

```python
from threej_cosmo import get_available_backends, coupling_matrix_TT

print(get_available_backends())  # ['cpu'] or ['cpu', 'gpu']

K_TT = coupling_matrix_TT(window_cls, lmax=2000, backend="gpu")
```

## Standalone C Compilation

The underlying C implementations can be compiled standalone for use outside Python:

**CPU (requires GCC with OpenMP):**
```bash
cd src
make        # Builds thrj_000 and thrj_220
make info   # Shows compiler and flags being used
```

**GPU (requires NVIDIA HPC SDK):**
```bash
cd src_gpu
make        # Builds thrj_000_gpu and thrj_220_gpu
```

Adjust `-gpu=cc80` in `src_gpu/Makefile` for your GPU architecture (cc70 for V100, cc80 for A100, cc90 for H100).

## Citation

If you use this code, please cite:

```bibtex
@article{kiddier2026threej,
  author = {Kiddier, Georgia and Gratton, Steven},
  title = {Fast computation of temperature and polarization coupling matrices},
  year = {2026}
}
```

## License

MIT
