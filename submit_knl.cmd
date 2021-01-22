#!/bin/bash -l
#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH -S 4 ###number of specialized cores per node (use 4 to get 64 cores per node for GENE)
#SBATCH -C knl,quad,cache  ###default setting on Cori, means 16GB fast RAM will be used as L3 cache; leaving 96GB per node
#SBATCH -t 00:30:00
#SBATCH -o python_MPI.out
#SBATCH -e python_MPI.err

module load python
srun -n 64 -c  1 python /global/u1/m/maxcurie/max/scripts/max_mode_finder/0MTMDispersion.py
