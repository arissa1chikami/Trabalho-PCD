
/* kmeans_1d_omp.c
   K-means 1D paralelo com OpenMP
   Compilação:
       gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
   Execução:
       ./kmeans_1d_omp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <ctype.h>

/* ---------- utilitários CSV 1D ---------- */
static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows = 0; char line[8192];
    while(fgets(line,sizeof(line),f)) {
        int only_ws = 1;
        for(char *p=line; *p; p++){
            if(!isspace((unsigned char)*p)){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out) {
    int R = count_rows(path);
    double *A = malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memória\n"); exit(1); }

    FILE *f = fopen(path,"r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    char line[8192]; int r=0;
    while(fgets(line,sizeof(line),f)){
        char *end;
        double v = strtod(line,&end);
        if(end!=line) A[r++] = v;
    }
    fclose(f);
    *n_out = r;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    FILE *f = fopen(path,"w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n",path); return; }
    for(int i=0;i<N;i++) fprintf(f,"%d\n",assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    FILE *f = fopen(path,"w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n",path); return; }
    for(int c=0;c<K;c++) fprintf(f,"%.6f\n",C[c]);
    fclose(f);
}

/* ---------- etapas do K-means ---------- */

/* Assignment: paralelizar loop i */
static double assignment_step_1d(const double *X, const double *C,
                                 int *assign, int N, int K)
{
    double sse = 0.0;
    #pragma omp parallel for reduction(+:sse) schedule(dynamic,100000)
    for(int i=0;i<N;i++){
        int best = 0;
        double bestd = (X[i]-C[0])*(X[i]-C[0]);
        for(int c=1;c<K;c++){
            double diff = X[i]-C[c];
            double d = diff*diff;
            if(d<bestd){ bestd=d; best=c; }
        }
        assign[i]=best;
        sse += bestd;
    }
    return sse;
}

/* Update: opção A – acumuladores locais por thread e redução */
static void update_step_1d(const double *X, double *C, const int *assign, int N, int K)
{
    int nthreads = omp_get_max_threads();
    double **sum_thread = malloc(nthreads * sizeof(double*));
    int    **cnt_thread = malloc(nthreads * sizeof(int*));
    for(int t=0;t<nthreads;t++){
        sum_thread[t] = calloc(K,sizeof(double));
        cnt_thread[t] = calloc(K,sizeof(int));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic,100000)
        for(int i=0;i<N;i++){
            int a = assign[i];
            cnt_thread[tid][a]++;
            sum_thread[tid][a] += X[i];
        }
    }

    // redução final
    for(int c=0;c<K;c++){
        double sum=0.0; int cnt=0;
        for(int t=0;t<nthreads;t++){
            sum += sum_thread[t][c];
            cnt += cnt_thread[t][c];
        }
        if(cnt>0) C[c] = sum/cnt;
        else      C[c] = X[0];
    }

    for(int t=0;t<nthreads;t++){ free(sum_thread[t]); free(cnt_thread[t]); }
    free(sum_thread); free(cnt_thread);
}

/* ---------- função principal do K-means ---------- */
static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d(X, C, assign, N, K);
        double rel = fabs(sse - prev_sse) / (prev_sse>0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }
        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4]\n", argv[0]);
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;

    int N=0,K=0;
    double *X = read_csv_1col(pathX,&N);
    double *C = read_csv_1col(pathC,&K);
    int *assign = malloc(N*sizeof(int));

    double t0 = omp_get_wtime();
    int iters=0; double sse=0.0;
    kmeans_1d(X,C,assign,N,K,max_iter,eps,&iters,&sse);
    double t1 = omp_get_wtime();

    printf("K-means 1D (OpenMP)\n");
    printf("N=%d K=%d Threads=%d\n", N,K,omp_get_max_threads());
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.3f ms\n",
           iters,sse,(t1-t0)*1000.0);

    write_assign_csv("assign.csv",assign,N);
    write_centroids_csv("centroids.csv",C,K);

    free(assign); free(X); free(C);
    return 0;
}