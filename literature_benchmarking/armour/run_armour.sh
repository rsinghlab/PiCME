#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1  
 
# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 4 CPU core(s)
#SBATCH -n 16
# Memory is evenly distributed amongst number of cores

#SBATCH --mem=50G
#SBATCH -t 96:00:00

## Provide a job name
#SBATCH -J run_armour
#SBATCH -o slurm_out/armour_%j.out
#SBATCH -e slurm_out/armour_%j.err

# source <your_environment_path>

which python
python3 run_model.py