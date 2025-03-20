#!/bin/bash

# Activate Conda environment
source /scratch/muhammadnawfal.cse.nitt/miniconda3/bin/activate
conda activate fyp

# Define delay in seconds
DELAY=1  # Adjust delay as needed (e.g., 5 seconds)

# Loop over values of L_s and L_g (both equal)
for L in 2 4; do
    # Loop over values of D_s and D_g such that D_s + D_g = 12
    for D_s in 2 4 6 8 10; do
        D_g=$((12 - D_s))  # Compute D_g
        
        # Loop over the vision_deep_replace_method values
        for method in "replace" "accumulate"; do
            echo "Submitting job: L_s=L_g=$L, D_s=$D_s, D_g=$D_g, method=$method"
            
            # Submit the job using sbatch, not srun, to avoid issues with job IDs
            sbatch --nodes=1 --gres=gpu:1 --cpus-per-task=4 --partition=gpu --exclusive \
                --wrap "python train.py --D_s $D_s --D_g $D_g --L_s $L --L_g $L --vision_deep_replace_method $method"
            
            # Wait for a specified delay before submitting the next job
            sleep $DELAY
        done
    done
done

echo "All jobs submitted with a delay!"
exit 0  # Exit script immediately

