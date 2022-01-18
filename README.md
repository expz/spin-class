# Spin Class

_Spinning up as a student of reinforcement learning._

This is based on advice and algorithms explained by OpenAI at [Spinning Up](https://spinningup.openai.com/en/latest/) although the code is my own except where otherwise indicated in comments.

This code uses [Weights and Biases](https://wandb.ai/site) to track experiments.

## Install

### Prerequisites

* Ubuntu 18.04 (presumably works on 20.04 and 22.04)
* Python 3.8
* pip
* virtualenv
* Nvidia GPU

### Install Mujoco

Mujoco is required for the InvertedPendulum-v2 and HalfCheetah-v2 environments.

#### Install prerequisites for Mujoco

```bash
sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libosmesa6-dev \
    net-tools \
    software-properties-common \
    xpra \
    xserver-org-dev

sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
  && sudo chmod a+x /usr/local/bin/patchelf
```

#### Install Mujoco library

```bash
mkdir -p ~/.mujoco

wget -o ~/.mujoco/mjkey.txt https://roboti.us/file/mjkey.txt
wget -o ~/.mujoco/mjpro150_linux.zip https://roboti.us/download/mjpro150_linux.zip
cd ~/.mujoco && unzip ./mjpro150_linux.zip

# libglewosmesa.so from Mujoco 1.5 is incompatible with Python >= 3.7,
 # so get a new version of the library from Mujoco 2.1.0
wget -o ~/.mujoco/mujoco210-linux-x86_64.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
cd ~/.mujoco && tar -xvzf ./mujoco210-linux-x86_64.tar.gz
cd ~/.mujoco/mjpro150/bin \
  && mv libglewosmesa.so libglewosmesa.old.so \
  && cp ~/.mujoco/mujoco210/bin/libglewosmesa.so . \
  && chmod 775 libglewosmesa.so
```

Set the library path and update bash configuration to automatically load library path.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mjpro150:/usr/lib/nvidia
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mjpro150:/usr/lib/nvidia' >> ~/.bashrc
```

### Set up virtual environment

From the directory where you want this repository to go, run:

```bash
git clone https://github.com/expz/spin-class.git
cd spin-class
virtualenv --python=/usr/bin/python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

(Note that it doesn't work to run these commands and then move the directory later, because the virtual environment is tied to the exact full path.)

### Log into Weights and Biases

This requires a free account with Weights and Biases.

```bash
wandb login
```

## Algorithms

### Vanilla Policy Gradient

This is implemented for CartPole-v0 and InvertedPendulum-v2.

#### Run reproducibility test

This runs the same algorithm with the same settings with different random seeds to test how consistent its performance is. From the root directory of this repository run

```bash
source venv/bin/activate
python spin_class/vpg_cartpole_reproducibility.py
```

for CartPole-v0 or

```bash
source venv/bin/activate
python spin_class/vpg_invpen_reproducibility.py
```

for InvertedPendulum-v2.

#### Run a hyperparameter search ("sweep")

From the Weights and Biases UI, create a new sweep in the desired project. Copy and paste the contents of `spin_class/vpg_cartpole_sweep.yaml` or `spin_class/vpg_invpen_sweep.yaml`  for cartpole or inverted pendulum environments respctively into the settings box and create the sweep.

Then for each search process you would like to start, from the root directory of this repository, run

```bash
source venv/bin/activate
```

and then copy, paste and run the sweep command from the Weights and Biases UI.
