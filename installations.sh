sudo apt-get update
python3 -m pip install --user --upgrade pip
python3 -m pip --version
python3 -m pip install --user virtualenv
python3 -m venv GNNEnv
source GNNEnv/bin/activate
pip3 install torch torchvision torchaudio
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install ogb
pip install wandb