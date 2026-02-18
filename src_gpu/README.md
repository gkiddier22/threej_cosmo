# GPU-Accelerated Coupling Matrix Computation

OpenMP target offloading versions for NVIDIA/AMD GPUs.

## Files

- `thrj_000_gpu.c` - Temperature (TT) coupling matrices
- `thrj_220_gpu.c` - Polarization (EE/EB) coupling matrices

## Compilation

### NVIDIA (nvc)
```bash
nvc -O3 -mp=gpu -gpu=cc80 -o thrj_000_gpu thrj_000_gpu.c -lm
nvc -O3 -mp=gpu -gpu=cc80 -o thrj_220_gpu thrj_220_gpu.c -lm
```

Adjust `-gpu=ccXX` for your GPU: `cc70` (V100), `cc80` (A100), `cc86` (RTX 30xx), `cc89` (RTX 40xx).

### AMD (amdclang)
```bash
amdclang -O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
    -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a \
    -o thrj_000_gpu thrj_000_gpu.c -lm
```

Adjust `-march=gfxXXX` for your GPU: `gfx908` (MI100), `gfx90a` (MI200).

## Usage

```bash
./thrj_000_gpu <power_spectrum.bin> <output.bin> <lmax> <write_output>
./thrj_220_gpu <power_spectrum.bin> <output.bin> <lmax> <pol> <write_output>
```

## Note

Apple Silicon (M1/M2/M3) does not support OpenMP GPU offloading. Use the CPU versions in `../src/` instead.
