#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* --- util CSV 1D: cada linha tem 1 número --- */
static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Erro ao abrir %s\n", path); exit(1); }
    int rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out) {
    int R = count_rows(path);
    if (R <= 0) { fprintf(stderr, "Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if (!A) { fprintf(stderr, "Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Erro ao abrir %s\n", path); free(A); exit(1); }

    char line[8192];
    int r = 0;
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (only_ws) continue;

        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if (!tok) {
            fprintf(stderr, "Linha %d sem valor em %s\n", r+1, path);
            free(A);
            exit(1);
        }
        A[r] = atof(tok);
        r++;
        if (r >= R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Erro ao abrir %s para escrita\n", path); return; }
    for (int i = 0; i < N; i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Erro ao abrir %s para escrita\n", path); return; }
    for (int c = 0; c < K; c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* --- Kernel CUDA para assignment --- */
__global__ void assignment_kernel(const double *X, const double *C, int *assign,
                                 double *sse_per_point, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        int best = -1;
        double bestd = 1e300;

        for (int c = 0; c < K; c++) {
            double diff = X[i] - C[c];
            double d = diff * diff;
            if (d < bestd) {
                bestd = d;
                best = c;
            }
        }

        assign[i] = best;
        sse_per_point[i] = bestd;
    }
}

/* --- Update no host --- */
static void update_step_1d(const double *X, double *C, const int *assign,
                          int N, int K) {
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));

    for (int i = 0; i < N; i++) {
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }

    for (int c = 0; c < K; c++) {
        if (cnt[c] > 0)
            C[c] = sum[c] / (double)cnt[c];
        else
            C[c] = X[0];
    }

    free(sum);
    free(cnt);
}

/* --- K-means CUDA --- */
static void kmeans_1d_cuda(const double *X_host, double *C_host, int *assign_host,
                          int N, int K, int max_iter, double eps,
                          int *iters_out, double *sse_out,
                          double *total_time_ms, int block_size)
{
    double *X_dev, *C_dev, *sse_per_point_dev;
    int *assign_dev;

    cudaMalloc(&X_dev, N * sizeof(double));
    cudaMalloc(&C_dev, K * sizeof(double));
    cudaMalloc(&assign_dev, N * sizeof(int));
    cudaMalloc(&sse_per_point_dev, N * sizeof(double));

    int grid_size = (N + block_size - 1) / block_size;

    /* --- TEMPOS INDIVIDUAIS --- */
    double time_h2d = 0.0;
    double time_d2h = 0.0;
    double time_kernel = 0.0;
    double time_update = 0.0;
    clock_t t0;

    /* --- TEMPO H2D --- */
    t0 = clock();
    cudaMemcpy(X_dev, X_host, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_dev, C_host, K * sizeof(double), cudaMemcpyHostToDevice);
    time_h2d += (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    double prev_sse = 1e300;
    double sse = 0;
    int it = 0;

    for (it = 0; it < max_iter; it++) {

        /* --- TEMPO DO KERNEL --- */
        t0 = clock();
        assignment_kernel<<<grid_size, block_size>>>(
            X_dev, C_dev, assign_dev, sse_per_point_dev, N, K
        );
        cudaDeviceSynchronize();
        time_kernel += (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC;

        double *sse_per_point_host = (double*)malloc(N * sizeof(double));

        /* --- TEMPO D2H --- */
        t0 = clock();
        cudaMemcpy(assign_host, assign_dev, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(sse_per_point_host, sse_per_point_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
        time_d2h += (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC;

        /* --- Calcula SSE total --- */
        sse = 0.0;
        for (int i = 0; i < N; i++)
            sse += sse_per_point_host[i];
        free(sse_per_point_host);

        /* --- Convergência --- */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0 ? prev_sse : 1);
        if (rel < eps) { it++; break; }

        /* --- Update step (CPU) ---*/
        t0 = clock();
        update_step_1d(X_host, C_host, assign_host, N, K);
        time_update += (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC;

        cudaMemcpy(C_dev, C_host, K * sizeof(double), cudaMemcpyHostToDevice);

        prev_sse = sse;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *total_time_ms = ms;

    *iters_out = it;
    *sse_out = sse;

    /* ---- PRINT TEMPOS DETALHADOS ---- */
    printf("\n=== TEMPOS DETALHADOS ===\n");
    printf("H2D: %.3f ms\n", time_h2d);
    printf("Kernel: %.3f ms\n", time_kernel);
    printf("D2H: %.3f ms\n", time_d2h);
    printf("Update (CPU): %.3f ms\n", time_update);
    printf("Total (CUDA event): %.3f ms\n", *total_time_ms);

    cudaFree(X_dev);
    cudaFree(C_dev);
    cudaFree(assign_dev);
    cudaFree(sse_per_point_dev);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/* --- Main --- */
int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Uso: %s dados.csv centroides.csv [iter] [eps] [block]\n", argv[0]);
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    int block_size = (argc > 5) ? atoi(argv[5]) : 256;
    const char *outAssign = (argc > 6) ? argv[6] : "assign_cuda.csv";
    const char *outCentroid = (argc > 7) ? argv[7] : "centroids_cuda.csv";

    int N, K;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc(N * sizeof(int));

    printf("=== K-means CUDA ===\n");
    printf("N=%d K=%d block=%d eps=%g\n", N, K, block_size, eps);

    int iters;
    double sse, total_ms;

    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps,
                   &iters, &sse, &total_ms, block_size);

    double throughput = (N * iters) / (total_ms / 1000.0);

    printf("\n=== RESULTADOS ===\n");
    printf("Iterações: %d\n", iters);
    printf("SSE final: %.6f\n", sse);
    printf("Tempo total: %.3f ms\n", total_ms);
    printf("Throughput: %.2f pontos/s\n", throughput);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    printf("\nArquivos gerados: %s , %s\n", outAssign, outCentroid);

    free(assign);
    free(X);
    free(C);
    return 0;
}
