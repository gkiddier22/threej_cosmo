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

int lmax;

static inline double calculate_threej_000_squared_direct(
    int J, int half_J1minus, int half_J2minus, int half_J3minus, int half_J,
    double* __restrict__ one_jp1,
    double* __restrict__ g,
    double* __restrict__ one_g) {

    return one_jp1[J] * g[half_J1minus] * g[half_J2minus] * g[half_J3minus] * one_g[half_J];
}


int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <lmax> <write_output (yes/no)>\n", argv[0]);
        return 1;
    }
    lmax = atoi(argv[3]);
    char *write_output = argv[4];
    int nlr;
    FILE *fpw;

    double *wl = (double*) calloc(2*lmax+1, sizeof(double));
    double *lnjp1 = (double*) calloc(4 * lmax + 1, sizeof(double));
    double *lng = (double*) calloc(2 * lmax + 1, sizeof(double));
    double *g = (double*) calloc(2 * lmax + 1, sizeof(double));
    double *one_g = (double*) calloc(2 * lmax + 1, sizeof(double));
    double *one_jp1 = (double*) calloc(4 * lmax + 1, sizeof(double));
    double *m = (double*) calloc((lmax+1)*(lmax+1), sizeof(double));

    fpw = fopen(argv[1], "rb");
    nlr = fread((void*)wl, sizeof(double), 2*lmax+1, fpw);
    if (nlr != (2*lmax+1)) {
        printf("Error reading power spectrum.\n");
        fclose(fpw);
        return -1;
    }
    fclose(fpw);

    for (int i = 0; i < 4 * lmax + 1; i++) {
        lnjp1[i] = log(i + 1.0);
        one_jp1[i] = 1.0 / (i + 1.0);
    }

    lng[0] = 0.0;
    one_g[0] = 1.0;
    for (int i = 1; i < 2 * lmax + 1; i++) {
        lng[i] = lng[i - 1] + log((i - 0.5) / i);
        one_g[i] = exp(-lng[i]);
    }

    for (int i = 0; i < 2 * lmax + 1; i++) {
        g[i] = exp(lng[i]);
    }

    double j3_sum, pref;
    int J, J1minus, J2minus, J3minus;
    const double scale_factor = M_1_PI * 0.25;

    double start = omp_get_wtime();
    #pragma omp parallel for private(j3_sum, J, J1minus, J2minus, J3minus, pref) \
    shared(m, wl, one_jp1, g, one_g) schedule(dynamic)
    for (int i = 0; i < lmax + 1; i++) {
        for (int k = i; k < lmax + 1; k++) {
            j3_sum = 0.;

            int l_start = k-i;
            int l_end = i + k + 1;

            int l_first = l_start;
            int J_first = i + k + l_first;
            if ((J_first & 1) == 1) {
                l_first++;
                J_first++;
            }

            if (l_first < l_end) {
                J = J_first;
                J1minus = -i + k + l_first;
                J2minus = i - k + l_first;
                J3minus = i + k - l_first;

                int half_J = J >> 1;
                int half_J1minus = J1minus >> 1;
                int half_J2minus = J2minus >> 1;
                int half_J3minus = J3minus >> 1;

                double threej_000_sq = calculate_threej_000_squared_direct(
                    J, half_J1minus, half_J2minus, half_J3minus, half_J,
                    one_jp1, g, one_g);
                pref = ((2 * l_first) + 1) * wl[l_first];
                j3_sum += pref * threej_000_sq;

                for (int l = l_first + 2; l < l_end; l += 2) {
                    J += 2;
                    half_J++;
                    half_J1minus++;
                    half_J2minus++;
                    half_J3minus--;

                    threej_000_sq = calculate_threej_000_squared_direct(
                        J, half_J1minus, half_J2minus, half_J3minus, half_J,
                        one_jp1, g, one_g);
                    pref = ((2 * l) + 1) * wl[l];
                    j3_sum += pref * threej_000_sq;
                }
            }

            m[i*(lmax+1)+k] = j3_sum * scale_factor;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lmax + 1; i++) {
        for (int j = i + 1; j < lmax + 1; j++) {
            m[j * (lmax + 1) + i] = m[i * (lmax + 1) + j];
        }
    }

    int size = (lmax + 1) * (lmax + 1);
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < size; idx++) {
        int j = idx % (lmax + 1);
        m[idx] *= (2. * j + 1.);
    }

    double end = omp_get_wtime();
    printf("Wall-clock time: %g seconds\n", end - start);

    if (strcmp(write_output, "yes") == 0) {
        FILE *fpm = fopen(argv[2], "wb");
        if (!fpm) {
            perror("Error opening output file");
            free(wl);
            free(m);
            return 1;
        }
        fwrite((void*)m, sizeof(double), (lmax+1)*(lmax+1), fpm);
        fclose(fpm);
        printf("Matrix written to %s.\n", argv[2]);
    }

    free(lnjp1);
    free(lng);
    free(g);
    free(one_g);
    free(one_jp1);
    free(wl);
    free(m);

    return 0;
}
