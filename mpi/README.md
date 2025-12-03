# Trabalho-PCD — K-Means 1D Paralelo
Implementação do algoritmo K-Means 1D com paralelização progressiva utilizando OpenMP.  
Projeto da disciplina de Programação Concorrente e Distribuída.

## Estrutura do projeto
serial/ → versão sequencial (baseline)  
openmp/ → versão paralela com OpenMP (CPU)  
cuda/ → versão paralela com CUDA (GPU)  
mpi/ → versão paralela com MPI  


## Requisitos
🔹 Hardware  
Um ou mais computadores conectados na mesma rede.  
Cada máquina deve possuir o MPI instalado.  

🔹 Software  
OpenMPI  
Compilador C (ex.: ```gcc```)  
Sistema operacional Linux (nativo ou WSL)  


## Compilação
```mpicc -o kmeans_1d_mpi kmeans_1d_mpi.c -lm```   


## Execução
1. Execução Local (uma máquina)  
```mpirun -np 4 ./kmeans_1d_mpi dados.csv centroides_iniciais.csv```  

2. Execução em Múltiplas Máquinas (Cluster MPI)  
Passo 1 — Configurar acesso SSH sem senha  
Na máquina principal:  
```ssh-keygen -t rsa```  
```ssh-copy-id usuario@IP_da_outra_maquina```  
Teste:  
```ssh usuario@IP_da_outra_maquina```  
Passo 2 — Criar o arquivo ```hosts.txt```  
Exemplo:  
```192.168.1.10 slots=4```  
```192.168.1.11 slots=4```  
Passo 3 — Executar no cluster  
```mpirun -np 4 -hostfile hosts.txt ./kmeans_1d_mpi dados.csv centroides_iniciais.csv```  


## Mudar quantidade de processos
// Quantidade de processos - 1  
```mpirun -np 1 -hostfile hosts.txt ./kmeans_1d_mpi dados.csv centroides_iniciais.csv```  

// Quantidade de processos - 2  
```mpirun -np 2 -hostfile hosts.txt ./kmeans_1d_mpi dados.csv centroides_iniciais.csv```  

// Quantidade de processos - 3  
```mpirun -np 3 -hostfile hosts.txt ./kmeans_1d_mpi dados.csv centroides_iniciais.csv```  

// Quantidade de processos - 4  
```mpirun -np 4 -hostfile hosts.txt ./kmeans_1d_mpi dados.csv centroides_iniciais.csv```  

## Resultados e métricas
SSE (Sum of Squared Errors)  
Tempo de execução (ms)  
Tempo operação Allreduce (ms)  
Speedup


## Grupo
-Arissa Yumi Chikami  
-Júlia Harue Katsurayama  
-Robert Angelo de Souza Santos  

## Disciplina

Programação Concorrente e Distribuída (PCD)  
Profs. Álvaro e Denise — Turma I  
Universidade Federal de São Paulo - Campus São José dos Campos  
