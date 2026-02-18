/*
 * thrj_000_gpu.c
 *
 * GPU-accelerated computation of CMB temperature coupling matrices
 * using OpenMP target offloading and incremental index arithmetic.
 *
 * Author: Georgia Kiddier
 * Date: 19-11-2025
 * License: MIT
 *
 * Compilation (NVIDIA):
 *     nvc -O3 -mp=gpu -gpu=cc80 -o thrj_000_gpu thrj_000_gpu.c -lm
 *
 * Compilation (AMD):
 *     amdclang -O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
 *         -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -o thrj_000_gpu thrj_000_gpu.c -lm
 *
 * Usage:
 *     ./thrj_000_gpu <power_spectrum_file> <output_matrix_file> <lmax> <write_output>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#ifndef M_1_PI
#define M_1_PI 0.31830988618379067154
#endif

int lmax;

#pragma omp declare target
static inline double calculate_threej_000_squared_direct(
    int J, int half_J1minus, int half_J2minus, int half_J3minus, int half_J,
    double* one_jp1, double* g, double* one_g) {

    return one_jp1[J] * g[half_J1minus] * g[half_J2minus] * g[half_J3minus] * one_g[half_J];
}
#pragma omp end declare target

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
    double *lng = (double*) calloc(2*lmax+1, sizeof(double));
    double *g = (double*) calloc(2*lmax+1, sizeof(double));
    double *one_g = (double*) calloc(2*lmax+1, sizeof(double));
    double *one_jp1 = (double*) calloc(4*lmax+1, sizeof(double));
    double *m = (double*) calloc((lmax+1)*(lmax+1), sizeof(double));

    if (!wl || !lng || !g || !one_g || !one_jp1 || !m) {
        fprintf(stderr, "Memory allocation failed!\n");
        return -1;
    }

    fpw = fopen(argv[1], "rb");
    if (!fpw) {
        fprintf(stderr, "Cannot open file %s\n", argv[1]);
        return -1;
    }

    nlr = fread((void*)wl, sizeof(double), 2*lmax+1, fpw);
    if (nlr != (2*lmax+1)) {
        fprintf(stderr, "Error reading power spectrum\n");
        fclose(fpw);
        return -1;
    }
    fclose(fpw);

    for (int i = 0; i < 4*lmax+1; i++) {
        one_jp1[i] = 1.0 / (i + 1.0);
    }

    lng[0] = 0.0;
    one_g[0] = 1.0;
    for (int i = 1; i < 2*lmax+1; i++) {
        lng[i] = lng[i-1] + log((i - 0.5) / i);
        one_g[i] = exp(-lng[i]);
    }

    for (int i = 0; i < 2*lmax+1; i++) {
        g[i] = exp(lng[i]);
    }

    const double scale_factor = M_1_PI * 0.25;
    const int LMAX = lmax;
    const int n_wl = 2*LMAX + 1;
    const int n_g = 2*LMAX + 1;
    const int n_onejp1 = 4*LMAX + 1;
    const int n_m = (LMAX + 1) * (LMAX + 1);

    double start = omp_get_wtime();

    // Calculate total number of (i,k) pairs in upper triangle
    // i ranges from 0 to LMAX, k ranges from i to LMAX
    // Total pairs = sum_{i=0}^{LMAX} (LMAX - i + 1) = N*(N+1)/2
    const int N = LMAX + 1;
    const long n_pairs = ((long)N * (long)(N + 1)) / 2;

    #pragma omp target data \
      map(to: wl[0:n_wl], one_jp1[0:n_onejp1], g[0:n_g], one_g[0:n_g], LMAX, N, n_pairs) \
      map(from: m[0:n_m])
    {
        // Main computation kernel - linearized triangular iteration
        #pragma omp target teams distribute parallel for
        for (long idx = 0; idx < n_pairs; idx++) {
            // Convert linear index to triangular (i, k) coordinates
            // Row i (starting from i=0) has (N - i) elements
            // Cumulative elements before row i: sum_{r=0}^{i-1}(N-r) = i*(2N-i+1)/2
            // We solve for i given idx using the inverse formula

            // Using quadratic formula to find i:
            // idx = i*(2N-i+1)/2 + (k-i)
            // For the row start: idx_row = i*(2N-i+1)/2
            // Rearranging: i^2 - (2N+1)*i + 2*idx_row = 0
            // But simpler: find i such that row_start(i) <= idx < row_start(i+1)

            double a = -0.5;
            double b = (double)N + 0.5;
            double c = -(double)idx;
            double discriminant = b * b - 4.0 * a * c;
            int i = (int)((-b + sqrt(discriminant)) / (2.0 * a));

            // Clamp to valid range
            if (i < 0) i = 0;
            if (i > LMAX) i = LMAX;

            // Calculate row start for this i
            long row_start = ((long)i * (long)(2 * N - i + 1)) / 2;

            // Adjust i if we overshot
            while (i > 0 && row_start > idx) {
                i--;
                row_start = ((long)i * (long)(2 * N - i + 1)) / 2;
            }
            while (i < LMAX && row_start + (N - i) <= idx) {
                i++;
                row_start = ((long)i * (long)(2 * N - i + 1)) / 2;
            }

            // Calculate k from offset within row
            int k = i + (int)(idx - row_start);

            // Now compute the coupling matrix element for (i, k)
            double j3_sum = 0.;

            int l_start = k-i;
            int l_end = i+k+1;

            int l_first = l_start;
            int J_first = i + k + l_first;
            if ((J_first & 1) == 1) {
                l_first++;
                J_first++;
            }

            if (l_first < l_end) {
                int J = J_first;
                int J1minus = -i + k + l_first;
                int J2minus = i - k + l_first;
                int J3minus = i + k - l_first;

                int half_J = J >> 1;
                int half_J1minus = J1minus >> 1;
                int half_J2minus = J2minus >> 1;
                int half_J3minus = J3minus >> 1;

                double threej_000_sq = calculate_threej_000_squared_direct(
                    J, half_J1minus, half_J2minus, half_J3minus, half_J,
                    one_jp1, g, one_g);
                double pref = ((2 * l_first) + 1) * wl[l_first];
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

            m[i*(LMAX+1)+k] = j3_sum * scale_factor;
        }

        // Symmetry copy - linearized for full parallelization
        // Number of off-diagonal pairs: N*(N-1)/2
        const long n_sym = ((long)N * (long)(N - 1)) / 2;
        #pragma omp target teams distribute parallel for
        for (long idx = 0; idx < n_sym; idx++) {
            // Convert linear index to (i, j) where j > i
            double a_sym = -0.5;
            double b_sym = (double)N - 0.5;
            double c_sym = -(double)idx;
            double disc_sym = b_sym * b_sym - 4.0 * a_sym * c_sym;
            int i = (int)((-b_sym + sqrt(disc_sym)) / (2.0 * a_sym));
            if (i < 0) i = 0;
            if (i >= N) i = N - 1;
            long row_start_sym = ((long)i * (long)(2 * N - i - 1)) / 2;
            while (i > 0 && row_start_sym > idx) {
                i--;
                row_start_sym = ((long)i * (long)(2 * N - i - 1)) / 2;
            }
            while (i < N - 1 && row_start_sym + (N - i - 1) <= idx) {
                i++;
                row_start_sym = ((long)i * (long)(2 * N - i - 1)) / 2;
            }
            int j = i + 1 + (int)(idx - row_start_sym);
            m[j * N + i] = m[i * N + j];
        }

        // Multiply by (2j+1) - fully linearized
        const long n_total = (long)N * (long)N;
        #pragma omp target teams distribute parallel for
        for (long idx = 0; idx < n_total; idx++) {
            int j = (int)(idx % N);
            m[idx] *= (2. * j + 1.);
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

    free(lng);
    free(g);
    free(one_g);
    free(one_jp1);
    free(wl);
    free(m);

    return 0;
}
