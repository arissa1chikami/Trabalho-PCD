//mpirun -np 1 ./kmeans_1d_mpi dados.csv centroides_iniciais.csv
// cd /mnt/c/PCD/Trabalho-PCD/MPI

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---------------------------------------------------
   Função que lê CSV simples (1 coluna)
   ---------------------------------------------------*/
double* read_csv_1col(const char *path, int *n_out, int rank)
{
    double *A = NULL;
    int R = 0;

    if (rank == 0) {
        FILE *f = fopen(path, "r");
        if (!f) { printf("Erro ao abrir %s\n", path); MPI_Abort(MPI_COMM_WORLD,1); }

        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (line[0] != '\n' && line[0] != '\r')
                R++;
        }
        rewind(f);

        A = malloc(R * sizeof(double));
        for (int i = 0; i < R; i++) {
            if (!fgets(line, sizeof(line), f)) break;
            A[i] = atof(line);
        }
        fclose(f);
    }

    MPI_Bcast(&R, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        A = malloc(R * sizeof(double));

    MPI_Bcast(A, R, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    *n_out = R;
    return A;
}

/* ---------------------------------------------------
   Escrever outputs do sequencial (apenas rank 0)
   ---------------------------------------------------*/
void write_assign_csv(const char *path, int *assign, int N)
{
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

void write_centroids_csv(const char *path, double *C, int K)
{
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int i = 0; i < K; i++)
        fprintf(f, "%.6f\n", C[i]);
    fclose(f);
}

/* ---------------------------------------------------
   K-means MPI
   ---------------------------------------------------*/
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 3) {
        if (rank == 0)
            printf("Uso: mpirun -np P ./kmeans_1d_mpi dados.csv centroides.csv\n");
        MPI_Finalize();
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];

    const char *assign_out = "assign_mpi.csv";
    const char *cent_out   = "centroids_mpi.csv";

    int max_iter = 50;
    double eps = 1e-4;

    int N, K;
    double *X = read_csv_1col(pathX, &N, rank);
    double *C = read_csv_1col(pathC, &K, rank);

    int chunk = N / P;
    int ini = rank * chunk;
    int fim = (rank == P-1) ? N : ini + chunk;
    int localN = fim - ini;

    int *assign_local = malloc(localN * sizeof(int));
    int *assign_full  = NULL;
    if (rank == 0) assign_full = malloc(N * sizeof(int));

    double *sum_local  = malloc(K * sizeof(double));
    double *sum_global = malloc(K * sizeof(double));
    int *cnt_local     = malloc(K * sizeof(int));
    int *cnt_global    = malloc(K * sizeof(int));

    double prev_sse = 1e300;
    int iters = 0;

    double t_global_start = MPI_Wtime();

    double total_comm = 0.0;
    double time_allreduce = 0.0;
    double time_reduce = 0.0;
    double time_bcast = 0.0;

    for (int it = 0; it < max_iter; it++) {
        iters = it + 1;

        for (int c = 0; c < K; c++) {
            sum_local[c] = 0.0;
            cnt_local[c] = 0;
        }

        double sse_local = 0.0;

        for (int i = 0; i < localN; i++) {
            double xi = X[ini + i];

            int best = 0;
            double bestd = (xi - C[0]) * (xi - C[0]);

            for (int c = 1; c < K; c++) {
                double d = (xi - C[c]) * (xi - C[c]);
                if (d < bestd) { bestd = d; best = c; }
            }

            assign_local[i] = best;
            sse_local += bestd;
            sum_local[best] += xi;
            cnt_local[best] += 1;
        }

        /* REDUÇÕES MPI */
        double t0 = MPI_Wtime();
        double sse_global = 0.0;
        MPI_Reduce(&sse_local, &sse_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        time_reduce += (MPI_Wtime() - t0);

        t0 = MPI_Wtime();
        MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        double t_ar = MPI_Wtime() - t0;
        time_allreduce += t_ar;

        if (rank == 0) {
            for (int c = 0; c < K; c++) {
                if (cnt_global[c] > 0)
                    C[c] = sum_global[c] / cnt_global[c];
                else
                    C[c] = X[0];
            }

            double rel = fabs(sse_global - prev_sse) / prev_sse;
            if (rel < eps) break;
            prev_sse = sse_global;
        }

        t0 = MPI_Wtime();
        MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        time_bcast += (MPI_Wtime() - t0);
    }

    /* ---------------- GATHER assignments (corrigido) ---------------- */
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recvcounts = malloc(P * sizeof(int));
        displs = malloc(P * sizeof(int));
    }

    int local_count = localN;

    /* Cada processo envia seu tamanho */
    MPI_Gather(&local_count, 1, MPI_INT,
               recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Rank 0 calcula deslocamentos */
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < P; i++)
            displs[i] = displs[i-1] + recvcounts[i-1];
    }

    /* Gatherv agora funciona com blocos de tamanho diferente */
    MPI_Gatherv(assign_local, local_count, MPI_INT,
                assign_full, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }

    double t_total = MPI_Wtime() - t_global_start;
    total_comm = time_reduce + time_allreduce + time_bcast;

    /* PRINTS IGUAIS AO SEQUENCIAL */
    if (rank == 0) {
        printf("K-means 1D (MPI)\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
        printf("Iterações: %d | SSE final: %.6f | Tempo total: %.3f ms\n",
               iters, prev_sse, t_total * 1000.0);

        printf("\n--- TEMPOS DE COMUNICAÇÃO ---\n");
        printf("MPI_Reduce:    %.6f s\n", time_reduce);
        printf("MPI_Allreduce: %.6f s\n", time_allreduce);
        printf("MPI_Bcast:     %.6f s\n", time_bcast);
        printf("Total comunicação: %.6f s (%.2f%% do tempo)\n",
               total_comm, 100.0 * total_comm / t_total);

        printf("\nCentroides finais:\n");
        for (int c = 0; c < K; c++) printf("C[%d] = %.6f\n", c, C[c]);

        write_assign_csv(assign_out, assign_full, N);
        write_centroids_csv(cent_out, C, K);
    }

    free(assign_local);
    free(sum_local);
    free(sum_global);
    free(cnt_local);
    free(cnt_global);
    free(X);
    free(C);
    if (rank == 0) free(assign_full);

    MPI_Finalize();
    return 0;
}
