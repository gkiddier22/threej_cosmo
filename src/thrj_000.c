/*
 * thrj_000.c
 *
 * Optimized computation of CMB temperature coupling matrices using
 * incremental index arithmetic and squared 3j symbols.
 *
 * Author: Georgia Kiddier
 * Date: 18-02-2025
 * License: MIT
 *
 * Compilation:
 *     gcc -O3 -march=native -fopenmp -o thrj_000 thrj_000.c -lm -ffast-math
 *
 * Usage:
 *     ./thrj_000 <power_spectrum_file> <output_matrix_file> <lmax> <write_output>
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
#else
#ifdef __AVX__
#define EXPLICIT_SIMD 1
#define VLEN 4
#define LOADU _mm256_loadu_pd
#define VZERO {0,0,0,0}
#else
#ifdef __SSE2__
#define EXPLICIT_SIMD 1
#define VLEN 2
#define LOADU _mm_loadu_pd
#define VZERO {0,0}
#endif
#endif
#endif

#ifdef EXPLICIT_SIMD
#include <immintrin.h>
typedef double Tv __attribute__ ((vector_size (VLEN*8)));
#endif

int main(int argc, const char *argv[]) {
  if (argc < 5) {
    fprintf(stderr, "Usage: %s <input_file> <output_file> <lmax> <write_output (yes/no)>\n", argv[0]);
    return 1;
  }
  int lmax = atoi(argv[3]);
  const char *write_output = argv[4];

  double *wl = (double*) calloc(2*lmax+1+VLEN, sizeof(double));
  FILE *fpw = fopen(argv[1], "rb");
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

  double *g = (double*) calloc(2*lmax+1+VLEN, sizeof(double));
  // factor needed inside the critical loop: 1/(g[i]*(2i+1))
  double *fct = (double*) calloc(2*lmax+1+VLEN, sizeof(double));
  for (int i=0; i<2*lmax+1+VLEN; i++) {
    g[i] = (i==0) ? 1. : g[i-1]*((i-0.5)/i);
    fct[i] = 1. / (g[i]*(2*i + 1.));
    // pre-multiply wl to save work in the critical loop
    wl[i] *= (2*i+1) * (M_1_PI * 0.25);
  }

  double *m = (double*) calloc((lmax+1)*(lmax+1), sizeof(double));
  double start = omp_get_wtime();

#ifdef EXPLICIT_SIMD

  #pragma omp parallel for schedule(dynamic)
  for (int i=0; i<=lmax; i++) {
    for (int k=i; k<=lmax; k+=VLEN) {
      Tv j3_sum = VZERO;
      for (int ofs=0; ofs<=i; ++ofs) {
        j3_sum += LOADU(&wl[k-i+2*ofs]) * LOADU(&fct[k+ofs]) * LOADU(&g[k-i+ofs]) * g[ofs] * g[i-ofs];
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
  for (int i=0; i<=lmax; i++) {
    for (int k=i; k<=lmax; ++k) {
      const double *wl_actual = ((k-i)&1) ? wl_odd : wl_even;
      int l0 = (k-i)/2;
      double j3_sum = 0.;
      for (int ofs=0; ofs<=i; ++ofs)
        j3_sum += wl_actual[l0+ofs] * fct[k+ofs] * g[k-i+ofs] * g[ofs] * g[i-ofs];

      m[i*(lmax+1)+k] = j3_sum * (2. * k + 1.);
      m[k*(lmax+1)+i] = j3_sum * (2. * i + 1.);
    }
  }

#endif

  printf("Wall-clock time: %g seconds\n", omp_get_wtime() - start);

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
