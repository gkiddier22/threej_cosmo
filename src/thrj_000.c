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

int main(int argc, const char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <lmax> <write_output (yes/no)>\n", argv[0]);
        return 1;
    }
    int lmax = atoi(argv[3]);
    const char *write_output = argv[4];

    double *wl = (double*) calloc(2*lmax+1, sizeof(double));
    FILE *fpw = fopen(argv[1], "rb");
    int nlr = fread((void*)wl, sizeof(double), 2*lmax+1, fpw);
    if (nlr != (2*lmax+1)) {
        printf("Error reading power spectrum.\n");
        fclose(fpw);
        return -1;
    }
    fclose(fpw);

    // pre-multiply wl to save work in the critical loop
    for (int l = 0; l < 2 * lmax + 1; l++)
      wl[l] *= (2*l+1) * (M_1_PI * 0.25);

    // split wl into even and odd indices, to improve memory accesses
    double *wl_even = (double*) calloc(lmax+1, sizeof(double));
    for (int l = 0; l <= lmax; l++)
      wl_even[l] = wl[2*l];
    double *wl_odd = (double*) calloc(lmax, sizeof(double));
    for (int l = 0; l < lmax; l++)
      wl_odd[l] = wl[2*l+1];

    double *g = (double*) calloc(2 * lmax + 1, sizeof(double));
    double lng = 0.0;
    g[0] = 1.;
    for (int i = 1; i < 2 * lmax + 1; i++) {
        lng += + log((i - 0.5) / i);
        g[i] = exp(lng);
    }
    // factor needed inside the critical loop: 1/(g[i]*(2i+1))
    double *fct = (double*) calloc(2 * lmax + 1, sizeof(double));
    for (int i=0; i<2*lmax+1; i++)
        fct[i] = 1. / (g[i]*(2*i + 1.));

    double *m = (double*) calloc((lmax+1)*(lmax+1), sizeof(double));
    double start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<=lmax; i++) {
        for (int k = i; k <= lmax; ++k) {
            const double *wl_actual = ((k-i)&1) ? wl_odd : wl_even;
            int l0 = (k-i)/2;
            double j3_sum = 0.;
            for (int ofs=0; ofs<=i; ++ofs)
                j3_sum += wl_actual[l0+ofs] * fct[k+ofs] * g[k-i+ofs] * g[ofs] * g[i-ofs];

            m[i*(lmax+1)+k] = j3_sum * (2. * k + 1.);
            m[k*(lmax+1)+i] = j3_sum * (2. * i + 1.);
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
