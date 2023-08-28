python3 -m venv pygenv
source pygenv/bin/activate
pip install ogb
pip install wandb
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pip3 install torch_scatter torch-sparse  torch-geometric torch-quiver torchmetrics