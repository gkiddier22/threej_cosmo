/*
 * thrj_220.c
 *
 * Optimized computation of CMB polarization coupling matrices using
 * incremental index arithmetic and squared 3j symbols.
 *
 * Author: Georgia Kiddier
 * Date: 18-02-2025
 * License: MIT
 *
 * Compilation:
 *     gcc -O3 -march=native -fopenmp -o thrj_220 thrj_220.c -lm -ffast-math
 *
 * Usage:
 *     ./thrj_220 <power_spectrum_file> <output_matrix_file> <lmax> <pol> <write_output>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#undef EXPLICIT_SIMD
#define VLEN 0

#ifdef __AVX512F__
#define EXPLICIT_SIMD 1
#define VLEN 8
#define LOADU _mm512_loadu_pd
#define VZERO {0,0,0,0,0,0,0,0}
#define VIOTA {0,1,2,3,4,5,6,7}
#define VSQRT _mm512_sqrt_pd
#else
#ifdef __AVX__
#define EXPLICIT_SIMD 1
#define VLEN 4
#define LOADU _mm256_loadu_pd
#define VZERO {0,0,0,0}
#define VIOTA {0,1,2,3}
#define VSQRT _mm256_sqrt_pd
#else
#ifdef __SSE2__
#define EXPLICIT_SIMD 1
#define VLEN 2
#define LOADU _mm_loadu_pd
#define VZERO {0,0}
#define VIOTA {0,1}
#define VSQRT _mm_sqrt_pd
#endif
#endif
#endif

#ifdef EXPLICIT_SIMD
#include <immintrin.h>
typedef double Tv __attribute__ ((vector_size (VLEN*8)));
#endif

int main(int argc, char *argv[]) {
  if (argc < 5) {
    fprintf(stderr, "Usage: %s <input_file> <output_file> <lmax> <pol> <write_output (yes/no)>\n", argv[0]);
    return 1;
  }

  int lmax = atoi(argv[3]);
  char *pol = argv[4];
  char *write_output = argv[5];

  // Allocate arrays - use proper sizes
  double *wl = (double*) calloc(2*lmax+1+VLEN, sizeof(double));
  double *g = (double*) calloc(2*lmax+1+VLEN, sizeof(double));
  double *fct = (double*) calloc(2*lmax+1+VLEN, sizeof(double));
  double *m = (double*) calloc((lmax+1)*(lmax+1), sizeof(double));

  if (!wl || !g || !fct || !m) {
    fprintf(stderr, "Memory allocation failed!\n");
    return -1;
  }

  FILE *fpw = fopen(argv[1], "rb");
  if (!fpw) {
    fprintf(stderr, "Error opening input file.\n");
    return -1;
  }
  int nlr = fread((void*)wl, sizeof(double), 2*lmax+1, fpw);
  if (nlr != (2*lmax+1)) {
    printf("Error reading power spectrum.\n");
    fclose(fpw);
    return -1;
  }
  fclose(fpw);
  // safety zone for vectorization
  for (int i=2*lmax+1; i<2*lmax+1+VLEN; ++i)
    wl[i] = 0;

  if (strcmp(pol, "EE") != 0 && strcmp(pol, "EB") != 0) {
    fprintf(stderr, "Error: pol must be either 'EE' or 'EB'.\n");
    return 1;
  }

  // Precompute lookup tables
  double lng = 0.0;
  for (int i=0; i<2*lmax+1+VLEN; i++) {
    g[i] = exp(lng);
    fct[i] = 1./(g[i]*(2*i+1.));
    lng += log((i+0.5) / (i+1.));
    wl[i] *= (2*i+1) * (M_1_PI * 0.25);
  }

  double start = omp_get_wtime();

#ifdef EXPLICIT_SIMD

  #pragma omp parallel for schedule(dynamic)
  for (int i=2; i<=lmax; i++) {
    for (int k=i; k<=lmax; k+=VLEN) {
      Tv j3_sum = VZERO;
      Tv iota = VIOTA;
      Tv kv = (double)k + iota;
      int l0 = (k-i)/2;
      for (int ofs=0; ofs<=i; ++ofs) {
        Tv lmbda_sq = i*(i+1.)*(kv+1.)*(kv+2.);

        Tv lmbda2 = (kv+ofs+1.) * (2.*(i-ofs)+1.);

        Tv tmp = 1. + 2./i * (1. - lmbda2/((i+1.)*(kv+1.)));
        Tv A_sq = lmbda_sq * tmp*tmp;

        Tv pref_5_num_sq = 4.*lmbda2 * (2.*(kv-i+ofs+1) - 1.) * (kv-i+ofs+1) * ofs * (2*(kv+ofs)+3.) * (i-ofs+1.) * (2*ofs-1.);
        Tv B_sq = pref_5_num_sq / lmbda_sq;
        Tv threej_000_sq = g[i-ofs] * LOADU(&g[k-i+ofs]) * g[ofs] * LOADU(&fct[k+ofs]);
        Tv threej_000_2_sq = g[i-ofs+1] * LOADU(&g[k-i+ofs+1]) * g[ofs-1] * LOADU(&fct[k+ofs+1]);

        Tv inner_sq = A_sq * threej_000_sq - 2. * VSQRT(A_sq*B_sq*threej_000_sq * threej_000_2_sq) + B_sq * threej_000_2_sq;
        Tv eta_sq = ((i-1.)*(i+2.)*(kv-1.)*kv);
        j3_sum += LOADU(&wl[k-i+2*ofs]) * inner_sq / eta_sq;
      }
      for (size_t n=0; n<VLEN; ++n) {
        if (k+n<=lmax) {
          m[i*(lmax+1)+k+n] = j3_sum[n] * (2*(k+n)+1);
          m[(k+n)*(lmax+1)+i] = j3_sum[n] * (2*i+1);
        }
      }
    }
  }

#else

  // split wl into even and odd indices, to improve memory accesses
  double *wl_even = (double*) calloc(lmax+1, sizeof(double));
  for (int l = 0; l <= lmax; l++)
    wl_even[l] = wl[2*l];
  double *wl_odd = (double*) calloc(lmax, sizeof(double));
  for (int l = 0; l < lmax; l++)
    wl_odd[l] = wl[2*l+1];

  #pragma omp parallel for schedule(dynamic)
  for (int i=2; i<=lmax; i++) {
    for (int k=i; k<=lmax; k++) {
      double j3_sum = 0.0;

      const double *wl_actual = ((k-i)&1) ? wl_odd : wl_even;
      int l0 = (k-i)/2;
      for (int ofs=0; ofs<=i; ++ofs) {
        double lmbda_sq = i*(i+1.)*(k+1.)*(k+2.);

        double lmbda2 = (k+ofs+1.) * (2.*(i-ofs)+1.);

        double tmp = 1. + 2./i * (1. - lmbda2/((i+1.)*(k+1.)));
        double A_sq = lmbda_sq * tmp*tmp;

        double pref_5_num_sq = 4.*lmbda2 * (2.*(k-i+ofs+1) - 1.) * (k-i+ofs+1) * ofs * (2*(k+ofs)+3.) * (i-ofs+1.) * (2*ofs-1.);
        double B_sq = pref_5_num_sq / lmbda_sq;

        double threej_000_sq = g[i-ofs] * g[k-i+ofs] * g[ofs] * fct[k+ofs];
        double threej_000_2_sq = g[i-ofs+1] * g[k-i+ofs+1] * g[ofs-1] * fct[k+ofs+1];

        double inner_sq = A_sq * threej_000_sq - 2. * sqrt(A_sq*B_sq*threej_000_sq * threej_000_2_sq) + B_sq * threej_000_2_sq;
        double eta_sq = ((i-1.)*(i+2.)*(k-1.)*k);
        j3_sum += wl_actual[l0+ofs] * inner_sq / eta_sq;
      }
      m[i * (lmax+1) + k] = j3_sum*(2*k+1);
      m[k * (lmax+1) + i] = j3_sum*(2*i+1);
    }
  }

#endif

  double end = omp_get_wtime();
  printf("Wall-clock time: %g seconds\n", end - start);

  if (strcmp(write_output, "yes") == 0) {
    FILE *fpm = fopen(argv[2], "wb");
    if (!fpm) {
        perror("Error opening output file");
        return 1;
    }
    fwrite((void*)m, sizeof(double), (lmax+1)*(lmax+1), fpm);
    fclose(fpm);
    printf("Matrix written to %s.\n", argv[2]);
  }
}
