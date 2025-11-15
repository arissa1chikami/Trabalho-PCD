# Trabalho-PCD — K-Means 1D Paralelo
Implementação do algoritmo K-Means 1D com paralelização progressiva utilizando OpenMP, e futuraente CUDA e MPI. 
Projeto da disciplina de Programação Concorrente e Distribuída.

## Estrutura do projeto
serial/ → versão sequencial (baseline)  
openmp/ → versão paralela com OpenMP (CPU)  
- (as versões CUDA e MPI serão adicionadas posteriormente)  

## Compilação e execução
🔹 Sequencial  
```bash```  
gcc -O2 -std=c99 serial/kmeans_1d_naive.c -o kmeans_1d_naive -lm  
./kmeans_1d_naive dados.csv centroides_iniciais.csv  

## Resultados e métricas
SSE (Sum of Squared Errors)  
Tempo total de execução (ms)  
Speedup, Eficiência e Throughput em cada abordagem  


## Grupo
-Arissa Yumi Chikami  
-Júlia Harue Katsurayama  
-Robert Angelo de Souza Santos  

## Disciplina

Programação Concorrente e Distribuída (PCD)  
Profs. Álvaro e Denise — Turmas I e N  
Universidade Federal de São Paulo - Campus São José dos 
Campos  



