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

inline double sqr(double arg) { return arg*arg; }

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <lmax> <pol> <write_output (yes/no)>\n", argv[0]);
        return 1;
    }

    int lmax = atoi(argv[3]);
    char *pol = argv[4];
    char *write_output = argv[5];

    // Allocate arrays - use proper sizes
    double *wl = (double*) calloc(2*lmax+1, sizeof(double));
    double *g = (double*) calloc(2 * lmax + 1, sizeof(double));
    double *one_g_special = (double*) calloc(2 * lmax + 1, sizeof(double));
    double *m = (double*) calloc((lmax+1)*(lmax+1), sizeof(double));

    if (!wl || !g || !one_g_special || !m) {
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

    if (strcmp(pol, "EE") != 0 && strcmp(pol, "EB") != 0) {
        fprintf(stderr, "Error: pol must be either 'EE' or 'EB'.\n");
        return 1;
    }

    // Precompute lookup tables

    double lng = 0.0;
    g[0] = 1.;
    one_g_special[0] = 1.0;
    for (int i = 1; i < 2 * lmax + 1; i++) {
        lng += log((i - 0.5) / i);
        g[i] = exp(lng);
        one_g_special[i] = exp(-lng)/(2*i+1.);
    }

    // Pre-scale wl array
    double scale_factor = M_1_PI * 0.25;
    for (int l = 0; l < 2*lmax+1; l++) {
        wl[l] *= (2 * l + 1) * scale_factor;
    }

    // split wl into even and odd indices, to improve memory accesses
    double *wl_even = (double*) calloc(lmax+1, sizeof(double));
    for (int l = 0; l <= lmax; l++)
      wl_even[l] = wl[2*l];
    double *wl_odd = (double*) calloc(lmax, sizeof(double));
    for (int l = 0; l < lmax; l++)
      wl_odd[l] = wl[2*l+1];

    const int LMAX = lmax;
    const int N = LMAX + 1;

    double start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 2; i < N; i++) {
        for (int k = i; k < N; k++) {
            double j3_sum = 0.0;

            const double *wl_actual = ((k-i)&1) ? wl_odd : wl_even;
            int l0 = (k-i)/2;
            for (int ofs=0; ofs<=i; ++ofs) {
                double lmbda_sq = i*(i+1.)*(k+1.)*(k+2.);

                double lmbda2 = (k+ofs+1.) * (2.*(i-ofs)+1.);

                double A_sq = lmbda_sq * sqr(1. + 2./i * (1. - lmbda2/((i+1.)*(k+1.))));

                double pref_5_num_sq = 4.*lmbda2 * (2.*(k-i+ofs) + 1.) * ofs * (2.*(k+ofs) + 3.) * (i+1.-ofs) * (k+1.-i+ofs) * (2.*ofs - 1.);
                double B_sq = pref_5_num_sq / lmbda_sq;

                double threej_000_sq = g[i-ofs] * g[k-i+ofs] * g[ofs] * one_g_special[k+ofs];
                double threej_000_2_sq = g[i-ofs+1] * g[k-i+ofs+1] * g[ofs-1] * one_g_special[k+ofs+1];

                double inner_sq = A_sq * threej_000_sq - 2. * sqrt(A_sq*B_sq*threej_000_sq * threej_000_2_sq) + B_sq * threej_000_2_sq;
                j3_sum += wl_actual[l0+ofs] * inner_sq / ((i-1.)*(i+2.)*(k-1.)*k);
            }
            m[i * N + k] = j3_sum*(2*k+1);
            m[k * N + i] = j3_sum*(2*i+1);
        }
    }

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
