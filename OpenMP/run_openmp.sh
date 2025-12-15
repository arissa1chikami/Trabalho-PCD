#!/bin/bash

# ===============================
# Arquivos de entrada
# ===============================
DADOS="dados.csv"
CENTROIDES="centroides_iniciais.csv"

# ===============================
# Compilação
# ===============================
echo "Compilando código OpenMP..."
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm

# ===============================
# Arquivo de resultados
# ===============================
RESULTS="resultados_openmp.txt"
echo "Resultados OpenMP - K-Means 1D" > "$RESULTS"
echo "=========================================" >> "$RESULTS"

# ===============================
# Parâmetros experimentais
# ===============================
SCHEDULES=("static" "dynamic")
CHUNKS=(1000 10000 100000)
THREADS_LIST=(1 2 3 4 5 6)
RUNS=3

# ===============================
# Execuções
# ===============================
for SCHED in "${SCHEDULES[@]}"
do
    for CHUNK in "${CHUNKS[@]}"
    do
        echo "" >> "$RESULTS"
        echo "Schedule: $SCHED | Chunk: $CHUNK" >> "$RESULTS"
        echo "-----------------------------------------" >> "$RESULTS"

        export OMP_SCHEDULE="$SCHED,$CHUNK"

        for THREADS in "${THREADS_LIST[@]}"
        do
            export OMP_NUM_THREADS=$THREADS

            for ((RUN=1; RUN<=RUNS; RUN++))
            do
                echo "Threads=$THREADS | Execução=$RUN" >> "$RESULTS"
                ./kmeans_1d_omp $DADOS $CENTROIDES >> "$RESULTS"
                echo "" >> "$RESULTS"
            done
        done
    done
done

echo "Execuções finalizadas com sucesso."
