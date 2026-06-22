#!/usr/bin/env python3
"""
Test script for polarization (EE) coupling matrix computation.

Author: Georgia Kiddier
License: MIT
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

from threej_cosmo import coupling_matrix_EE

# Configuration
LMAX = 5000
SPECTRUM_TYPE = "EE"

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

WINDOW_CL_FILE = DATA_DIR / "window_function_cl_lmax_10000.dat"
OUTPUT_FILE = RESULTS_DIR / f"coupling_{SPECTRUM_TYPE}_{LMAX}.bin"
PLOT_FILE = RESULTS_DIR / f"coupling_{SPECTRUM_TYPE}_{LMAX}.png"


def plot_coupling_matrix(matrix, lmax, save_path, spectrum_type):
    """Visualize the coupling matrix."""
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix range: [{matrix.min():.2e}, {matrix.max():.2e}]")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis", norm=LogNorm())

    ax.set_title(f"{spectrum_type} Coupling Matrix")
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
    print(f"Computing {SPECTRUM_TYPE} coupling matrix (lmax={LMAX})...")
    matrix = coupling_matrix_EE(window_cls, lmax=LMAX, verbose=True)

    # Save result
    matrix.tofile(OUTPUT_FILE)
    print(f"Matrix saved to {OUTPUT_FILE}")

    # Plot
    plot_coupling_matrix(matrix, LMAX, PLOT_FILE, SPECTRUM_TYPE)

    print("\nTest completed!")


if __name__ == "__main__":
    main()
