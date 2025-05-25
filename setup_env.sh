conda create -n curiosity python=3.10 -y
source activate curiosity
pip install --upgrade pip
pip install -e .
python -m ipykernel install --user --name curiosity --display-name "curiosity kernel (curiosity)"