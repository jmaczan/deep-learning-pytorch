<div align="center">

# 🌻 Deep Learning models implemented in PyTorch

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

A collection of deep learning models implemented in PyTorch and PyTorch Lightning for educational purposes

Hopefully each of them will get a dedicated blog post on my humble tech blog [maczan.pl](https://maczan.pl)

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/jmaczan/deep-learning-pytorch
cd deep-learning-pytorch

# [OPTIONAL] create conda environment
conda create -n deep-learning-pytorch python=3.9
conda activate deep-learning-pytorch

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/jmaczan/deep-learning-pytorch
cd deep-learning-pytorch

# create conda environment and install dependencies
conda env create -f environment.yaml -n deep-learning-pytorch

# activate conda environment
conda activate deep-learning-pytorch
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```