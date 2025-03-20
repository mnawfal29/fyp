#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=/home/muhammadnawfal.cse.nitt/output/job_%j.out
#SBATCH --cpus-per-task=4

echo "Allocated Gokul node: jobid:"
squeue -a | grep gok
echo "------------------------------------"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo $SLURM_JOB_NODELIST
scontrol show hostnames $SLURM_JOB_NODELIST

nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo $head_node
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

bash
pwd
conda init bash
source /scratch/muhammadnawfal.cse.nitt/miniconda3/bin/activate
conda activate fyp

srun python train.py --D_s 6 --D_g 6 --L_s 4 --L_g 4
