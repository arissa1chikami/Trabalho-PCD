# Trabalho-PCD — K-Means 1D Paralelo
Implementação do algoritmo K-Means 1D com paralelização progressiva utilizando OpenMP. 
Projeto da disciplina de Programação Concorrente e Distribuída.

## Estrutura do projeto
serial/ → versão sequencial (baseline)  
openmp/ → versão paralela com OpenMP (CPU)
cuda/ → versão paralela com CUDA (GPU) 
- (a versão MPI será adicionada posteriormente)  

## Compilação e execução
🔹 OpenMP  
```bash```  
gcc -O2 -fopenmp -std=c99 openmp/kmeans_1d_omp.c -o kmeans_1d_omp -lm  
export OMP_NUM_THREADS=4 ./kmeans_1d_omp dados.csv centroides_iniciais.csv    

## Mudar de static para dynamic
- **Static:** Cada thread recebe um bloco fixo de iterações no início.  
- **Dynamic:** Cada thread pega blocos de iterações conforme termina o anterior.   

// Antes (dynamic)  
#pragma omp parallel for reduction(+:sse) schedule(dynamic,100000)  
#pragma omp for schedule(dynamic,100000)  

// Depois (static)  
#pragma omp parallel for reduction(+:sse) schedule(static,100000)  
#pragma omp for schedule(static,100000)  

## Resultados e métricas
SSE (Sum of Squared Errors)  
Tempo total de execução (ms)  
Speedup e Eficiência em cada abordagem  


## Grupo
-Arissa Yumi Chikami  
-Júlia Harue Katsurayama  
-Robert Angelo de Souza Santos  

## Disciplina

Programação Concorrente e Distribuída (PCD)  
Profs. Álvaro e Denise — Turma I  
Universidade Federal de São Paulo - Campus São José dos 
Campos  



