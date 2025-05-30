#!/bin/bash
#SBATCH --job-name=joby-job        
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1          
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12             
#SBATCH --output=interactive.out    
#SBATCH --error=interactive.err     

source /home/isegard/anaconda3/bin/activate mario_rl         # change to your own path
python train.py --curiosity                                  # please add the arguments you want to use