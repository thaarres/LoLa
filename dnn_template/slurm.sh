#!/bin/sh
#SBATCH --nodes=1
#SBATCH --time=200
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --output=parallel_%j.out
#SBATCH --error=parallel_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

srun -u  ./run_daint.sh
