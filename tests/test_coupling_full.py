#!/usr/bin/env python3
"""
Test script for polarization (EE) coupling matrix computation.

Author: Georgia Kiddier
License: MIT
"""

import numpy as np
from pathlib import Path

import threej_cosmo
import ducc0

# number of threads to use for ducc
nthreads = 8

# Configuration
LMAX = 10000

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"

WINDOW_CL_FILE = DATA_DIR / "window_function_cl_lmax_10000.dat"


def get_ducc_result(spec, lmax, optype):
    from time import time
    t0 = time()
    matrix_ducc = ducc0.misc.experimental.coupling_matrix_rect_new(
        spec=spec.reshape((1,-1)),
        optype=(optype,),
        nthreads=nthreads,
        res=np.empty((1,lmax+1,lmax+1))).reshape((lmax+1,lmax+1))
    # ducc doesn't have the final multiplication by 2*el2 + 1 built in.
    matrix_ducc *= (np.arange(lmax+1)*2.+1).reshape((1,-1))
    print(f"ducc0 execution time: {time()-t0}s")
    return matrix_ducc


def main():
    # Load window Cls
    print(f"Loading window Cls from {WINDOW_CL_FILE}...")
    window_cls = np.fromfile(WINDOW_CL_FILE, dtype=np.float64)
    print(f"Loaded {len(window_cls)} Cls")
    print()

    print(f"Computing TT coupling matrix (lmax={LMAX})...")
    matrix = threej_cosmo.coupling_matrix_TT(window_cls, lmax=LMAX, verbose=True)
    matrix_ducc = get_ducc_result(window_cls, LMAX, 0)
    print("L2 difference:", ducc0.misc.l2error(matrix, matrix_ducc))
    print()

    print(f"Computing EE coupling matrix (lmax={LMAX})...")
    matrix = threej_cosmo.coupling_matrix_EE(window_cls, lmax=LMAX, verbose=True)
    matrix_ducc = get_ducc_result(window_cls, LMAX, 2)
    print("L2 difference:", ducc0.misc.l2error(matrix, matrix_ducc))
    print()

    print(f"Computing TE coupling matrix (lmax={LMAX})...")
    matrix = threej_cosmo.coupling_matrix_TE(window_cls, lmax=LMAX, verbose=True)
    matrix_ducc = get_ducc_result(window_cls, LMAX, 1)
    print("L2 difference:", ducc0.misc.l2error(matrix, matrix_ducc))
    print()

    print(f"Computing EB coupling matrix (lmax={LMAX})...")
    matrix = threej_cosmo.coupling_matrix_EB(window_cls, lmax=LMAX, verbose=True)
    matrix_ducc = get_ducc_result(window_cls, LMAX, 3)
    print("L2 difference:", ducc0.misc.l2error(matrix, matrix_ducc))


if __name__ == "__main__":
    main()
