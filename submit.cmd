#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH -p debug
#SBATCH --nodes=3
#SBATCH -t 00:10:00
#SBATCH -o python_MPI.out
#SBATCH -e python_MPI.err

module load python
srun -n 96 -c  3 python MPI_test.py
