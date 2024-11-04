#!/bin/bash

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python hdbscan apex psutil umap-learn seaborn matplotlib tqdm gap-stat -i https://mirrors.aliyun.com/pypi/simple/
