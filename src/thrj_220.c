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

int lmax;

static inline double calculate_threej_000_squared_direct(
    int J, int half_J1minus, int half_J2minus, int half_J3minus, int half_J,
    const double* __restrict__ one_jp1,
    const double* __restrict__ g,
    const double* __restrict__ one_g) {

    return one_jp1[J] * g[half_J1minus] * g[half_J2minus] * g[half_J3minus] * one_g[half_J];
}

static inline double calculate_threej_022_squared_direct(
    int j1, int j2, int j3, int J, int J1minus, int J2minus, int J3minus,
    int half_J, int half_J1minus, int half_J2minus, int half_J3minus,
    int half_Jp2, int half_J1minusp2, int half_J2minusp2, int half_J3minusm2,
    const double* __restrict__ one_jp1,
    const double* __restrict__ g,
    const double* __restrict__ one_g,
    const double* __restrict__ one_r_j) {

    // Precompute products to reduce operations
    double j2_j2p1 = j2 * (j2 + 1.0);
    double j3p1_j3p2 = (j3 + 1.0) * (j3 + 2.0);
    double lmbda_sq = j2_j2p1 * j3p1_j3p2;
    double lmbda = sqrt(lmbda_sq);
    double inv_lmbda = 1.0 / lmbda;

    int J2MP1 = J2minus + 1;
    int J3MM1 = J3minus - 1;

    double one_r = one_r_j[j2-1] * one_r_j[j2+2] * one_r_j[j3-1] * one_r_j[j3];
    double pref_1_sq = one_r * one_r;

    double pref_2 = 2.0 * lmbda * one_jp1[j2-1];
    double lmbda2 = (J + 2) * (J1minus + 1);
    double pref_3 = 1.0 - 0.5 * lmbda2 * one_jp1[j2] * one_jp1[j3];

    double A = lmbda + pref_2 * pref_3;

    double pref_5_num_sq = lmbda2 * J2MP1 * J3minus * (J + 3.0) * (J1minus + 2.0) * (J2minus + 2.0) * J3MM1;
    double B = 0.5 * sqrt(pref_5_num_sq) * inv_lmbda;

    double threej_000_sq = calculate_threej_000_squared_direct(
        J, half_J1minus, half_J2minus, half_J3minus, half_J,
        one_jp1, g, one_g);

    double threej_000_2_sq = calculate_threej_000_squared_direct(
        J + 2, half_J1minusp2, half_J2minusp2, half_J3minusm2, half_Jp2,
        one_jp1, g, one_g);

    double sign_product = ((J + 1) & 1) ? -1.0 : 1.0;

    // Compute A^2, B^2, and cross term
    double A_sq = A * A;
    double B_sq = B * B;
    double cross_term = 2.0 * A * B * sign_product * sqrt(threej_000_sq * threej_000_2_sq);

    double inner_sq = A_sq * threej_000_sq + cross_term + B_sq * threej_000_2_sq;

    return pref_1_sq * inner_sq;
}


int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <lmax> <pol> <write_output (yes/no)>\n", argv[0]);
        return 1;
    }

    lmax = atoi(argv[3]);
    int nlr;
    char *pol = argv[4];
    char *write_output = argv[5];
    FILE *fpw;

    // Allocate arrays - use proper sizes
    double *wl = (double*) calloc(2*lmax+1, sizeof(double));
    double *lng = (double*) calloc(2 * lmax + 1, sizeof(double));
    double *g = (double*) calloc(2 * lmax + 1, sizeof(double));
    double *one_g = (double*) calloc(2 * lmax + 1, sizeof(double));
    double *one_jp1 = (double*) calloc(4 * lmax + 1, sizeof(double));
    // Fix: one_r_j only needs indices up to lmax+2, not (2*lmax)^2
    double *one_r_j = (double*) calloc(lmax + 4, sizeof(double));
    double *m = (double*) calloc((lmax+1)*(lmax+1), sizeof(double));

    if (!wl || !lng || !g || !one_g || !one_jp1 || !one_r_j || !m) {
        fprintf(stderr, "Memory allocation failed!\n");
        return -1;
    }

    fpw = fopen(argv[1], "rb");
    if (!fpw) {
        fprintf(stderr, "Error opening input file.\n");
        return -1;
    }
    nlr = fread((void*)wl, sizeof(double), 2*lmax+1, fpw);
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
    for (int i = 0; i < 4 * lmax + 1; i++) {
        one_jp1[i] = 1.0 / (i + 1.0);
    }

    one_r_j[0] = 0.0;
    for (int i = 1; i < lmax + 4; i++) {
        one_r_j[i] = 1.0 / sqrt((double)i);
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

    // Pre-scale wl array
    double scale_factor = M_1_PI * 0.25;
    for (int l = 0; l < 2*lmax+1; l++) {
        wl[l] *= (2 * l + 1) * scale_factor;
    }

    const int LMAX = lmax;
    const int N = LMAX + 1;

    double start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 2; i < N; i++) {
        for (int k = i; k < N; k++) {
            double j3_sum = 0.0;
            int jbase = i + k;
            int l_start = k - i;  // Since k >= i
            int l_end = jbase + 1;
            int k_minus_i = k - i;

            int l_first = l_start;
            int J_first = jbase + l_first;
            if ((J_first & 1) == 1) {
                l_first++;
                J_first++;
            }

            if (l_first < l_end) {
                int J = J_first;
                int J1minus = jbase - l_first;
                int J2minus = l_first + k_minus_i;
                int J3minus = l_first - k_minus_i;

                int half_J = J >> 1;
                int half_J1minus = J1minus >> 1;
                int half_J2minus = J2minus >> 1;
                int half_J3minus = J3minus >> 1;

                int half_Jp2 = (J + 2) >> 1;
                int half_J1minusp2 = (J1minus + 2) >> 1;
                int half_J2minusp2 = (J2minus + 2) >> 1;
                int half_J3minusm2 = (J3minus - 2) >> 1;

                double threej_022_sq = calculate_threej_022_squared_direct(
                    l_first, i, k, J, J1minus, J2minus, J3minus,
                    half_J, half_J1minus, half_J2minus, half_J3minus,
                    half_Jp2, half_J1minusp2, half_J2minusp2, half_J3minusm2,
                    one_jp1, g, one_g, one_r_j);
                j3_sum += wl[l_first] * threej_022_sq;

                for (int l = l_first + 2; l < l_end; l += 2) {
                    J += 2;
                    J1minus -= 2;
                    J2minus += 2;
                    J3minus += 2;

                    half_J++;
                    half_J1minus--;
                    half_J2minus++;
                    half_J3minus++;

                    half_Jp2++;
                    half_J1minusp2--;
                    half_J2minusp2++;
                    half_J3minusm2++;

                    threej_022_sq = calculate_threej_022_squared_direct(
                        l, i, k, J, J1minus, J2minus, J3minus,
                        half_J, half_J1minus, half_J2minus, half_J3minus,
                        half_Jp2, half_J1minusp2, half_J2minusp2, half_J3minusm2,
                        one_jp1, g, one_g, one_r_j);
                    j3_sum += wl[l] * threej_022_sq;
                }
            }

            m[i * N + k] = j3_sum;
        }
    }

    // Copy upper triangle to lower triangle
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            m[j * N + i] = m[i * N + j];
        }
    }

    // Multiply by (2j+1)
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m[i * N + j] *= (2.0 * j + 1.0);
        }
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

    free(lng);
    free(g);
    free(one_g);
    free(one_jp1);
    free(one_r_j);
    free(wl);
    free(m);

    return 0;
}
