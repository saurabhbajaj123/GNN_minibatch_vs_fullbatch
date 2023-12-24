sudo apt-get update
sudo apt install python3-pip
sudo apt install python3.8-venv
python3 -m pip install --user --upgrade pip
python3 -m pip --version
python3 -m pip install --user virtualenv
python3 -m venv GNNEnv
source GNNEnv/bin/activate