#!/bin/bash

# Create and activate environment
conda create -n curiosity python=3.10 -y
source activate curiosity

# Upgrade pip
pip install --upgrade pip

# Clone ML-Agents repo and install all packages
#cd Pyramid
#git clone https://github.com/Unity-Technologies/ml-agents.git
#cd ml-agents
#pip install -e ./ml-agents-envs
#pip install -e ./ml-agents  # contains mlagents-learn

# Go back to your repo and install editable mode
#cd ..
#cd ..

pip install -e .

# Register kernel
python -m ipykernel install --user --name curiosity --display-name "curiosity kernel (curiosity)"
