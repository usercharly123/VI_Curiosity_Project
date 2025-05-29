#!/bin/bash
#SBATCH --job-name=joby-job        # Change as needed
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1          
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12              # Adjust CPU allocation if needed
#SBATCH --output=interactive.out    # Output log file
#SBATCH --error=interactive.err     # Error log file

source /home/isegard/anaconda3/bin/activate mario_rl
python train.py --tr_epochs 8

# --init_model --init_icm --curiosity --perturb --global_epochs --tr_epochs --batch_size --n_step --hidden_size --lr --gamma