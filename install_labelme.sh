#!/bin/bash 
# run with "source install_labelme.sh"

echo "Installing Python version 3.8..."

sudo apt-get install python3-venv 
sudo snap install python38 
python38 -m venv –without-pip labelme_env 
source labelme_env/bin/activate 

# Install pip 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# install labelme 
pip3 install labelme

# run labelme 
labelme 


