#!/usr/bin/env python3
"""
Test script for temperature (TT) coupling matrix computation.

Author: Georgia Kiddier
License: MIT
"""

import numpy as np
import matplotlib
#matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

from threej_cosmo import coupling_matrix_TT

# Configuration
LMAX = 5

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

WINDOW_CL_FILE = DATA_DIR / "window_function_cl_lmax_10000.dat"
OUTPUT_FILE = RESULTS_DIR / f"coupling_TT_{LMAX}.bin"
PLOT_FILE = RESULTS_DIR / f"coupling_TT_{LMAX}.png"


def plot_coupling_matrix(matrix, lmax, save_path):
    """Visualize the coupling matrix."""
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix range: [{matrix.min():.2e}, {matrix.max():.2e}]")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis", norm=LogNorm())

    ax.set_title("TT Coupling Matrix")
    ax.set_xlabel(r"$\ell_1$")
    ax.set_ylabel(r"$\ell_2$")
    plt.colorbar(im, ax=ax, label="Coupling strength")

    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load window Cls
    print(f"Loading window Cls from {WINDOW_CL_FILE}...")
    window_cls = np.fromfile(WINDOW_CL_FILE, dtype=np.float64)
    print(f"Loaded {len(window_cls)} Cls")

    # Compute coupling matrix using the package
    print(f"Computing TT coupling matrix (lmax={LMAX})...")
    from time import time
    t0 = time()
    matrix = coupling_matrix_TT(window_cls, lmax=LMAX, verbose=True)
    print(time()-t0)
    print(matrix)

    import ducc0
    matrix_ducc = np.empty((1,LMAX+1,LMAX+1),dtype=np.float64)
    t0=time()
    ducc0.misc.experimental.coupling_matrix_rect_new(window_cls.reshape((1,-1)), optype=(0,), nthreads=8, res=matrix_ducc)
    print("ducc time:",time()-t0)
    matrix_ducc = matrix_ducc[0]
    matrix_ducc *= (2*np.arange(LMAX+1)+1).reshape((1,-1))
    print(matrix_ducc)
    print(ducc0.misc.l2error(matrix_ducc,matrix))

    print(matrix.shape, matrix_ducc.shape)
    import matplotlib.pyplot as plt
    plt.imshow(matrix-matrix_ducc)
    plt.show()

    # Save result
    matrix.tofile(OUTPUT_FILE)
    print(f"Matrix saved to {OUTPUT_FILE}")

    # Plot
    plot_coupling_matrix(matrix, LMAX, PLOT_FILE)

    print("\nTest completed!")


if __name__ == "__main__":
    main()
