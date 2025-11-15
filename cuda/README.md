# Trabalho-PCD — K-Means 1D Paralelo
Implementação do algoritmo K-Means 1D com paralelização progressiva utilizando CUDA. 
Projeto da disciplina de Programação Concorrente e Distribuída.

## Estrutura do projeto
serial/ → versão sequencial (baseline)  
openmp/ → versão paralela com OpenMP (CPU)
cuda/ → versão paralela com CUDA (GPU) 
- (a versão MPI será adicionada posteriormente)  

## Configuração do ambiente (Google Colaboratory)
🔹 CUDA  
```bash```  
!pip install git+https://github.com/lesc-ufv/cad4u.git &> /dev/null
!git clone https://github.com/lesc-ufv/cad4u &> /dev/null
%load_ext plugin
!git clone https://github.com/NVIDIA/cuda-samples.git
%cd cuda-samples/Samples/1_Utilities/deviceQuery
!make

## Compilação e execução
🔹 CUDA  
```bash```  
!nvcc -arch=sm_75 -O2 kmeans_1d_cuda.cu -o kmeans_cuda -lm
!./kmeans_cuda dados.csv centroides_iniciais.csv 50 1e-6 1024 assign.csv centroids.csv   

## Mudar tamanho do bloco
```bash```  
// Tamanho do bloco - 64  
!./kmeans_cuda dados.csv centroides_iniciais.csv 50 1e-6 64 assign.csv centroids.csv

// Tamanho do bloco - 128  
!./kmeans_cuda dados.csv centroides_iniciais.csv 50 1e-6 128 assign.csv centroids.csv

// Tamanho do bloco - 256 
!./kmeans_cuda dados.csv centroides_iniciais.csv 50 1e-6 256 assign.csv centroids.csv

// Tamanho do bloco - 512 
!./kmeans_cuda dados.csv centroides_iniciais.csv 50 1e-6 512 assign.csv centroids.csv

// Tamanho do bloco - 1024 
!./kmeans_cuda dados.csv centroides_iniciais.csv 50 1e-6 1024 assign.csv centroids.csv

## Resultados e métricas
SSE (Sum of Squared Errors)  
Tempo total de execução (ms)  
Speedup e Throughput
Tempos H2D, D2H, kernel


## Grupo
-Arissa Yumi Chikami  
-Júlia Harue Katsurayama  
-Robert Angelo de Souza Santos  

## Disciplina

Programação Concorrente e Distribuída (PCD)  
Profs. Álvaro e Denise — Turma I  
Universidade Federal de São Paulo - Campus São José dos 
Campos  