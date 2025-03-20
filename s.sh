#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --output=/home/muhammadnawfal.cse.nitt/output/job_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Activate Conda environment
source /scratch/muhammadnawfal.cse.nitt/miniconda3/bin/activate
conda activate fyp

# Loop over values of L_s and L_g (both equal)
for L in 2 4; do
    # Loop over values of D_s and D_g such that D_s + D_g = 12
    for D_s in 2 4 6 8 10; do
        D_g=$((12 - D_s))  # Compute D_g
        
        # Loop over the vision_deep_replace_method values
        for method in "replace" "accumulate"; do
            echo "Running job: L_s=L_g=$L, D_s=$D_s, D_g=$D_g, method=$method"
            
            # Run job using srun sequentially
            srun --nodes=1 --gres=gpu:1 --cpus-per-task=4 --partition=gpu \
                python train.py --D_s "$D_s" --D_g "$D_g" --L_s "$L" --L_g "$L" \
                --vision_deep_replace_method "$method"
            
        done
    done
done

echo "All jobs completed!"

