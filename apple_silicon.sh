#!/bin/bash
conda create -y -n pydeep -c conda-forge python=3.9
conda run -n pydeep pip install --upgrade pip
echo "Tensorflow 설치"
conda run -n pydeep pip install tensorflow
conda run -n pydeep pip install tensorflow-metal
conda run -n pydeep pip install keras-tuner
conda run -n pydeep pip install keras
echo "필수 패키지 설치"
conda run -n pydeep pip install ipykernel
conda install -y -n pydeep -c conda-forge pandas scikit-learn matplotlib
