#!/bin/bash
# seongjoo@codebasic.io
# 코드베이직 (c) 2023

# Check VENV_NAME is set
if [ -z "$VENV_NAME" ]; then
    VENV_NAME=pydeep
fi
echo "파이썬 환경 명칭: $VENV_NAME"

conda create -y -n $VENV_NAME -c conda-forge python=3.9
echo "NumPy 설치 (BLAS 가속을 위해 빌드)"
# gcc 설치 확인
if ! command -v gcc &> /dev/null
then
    echo "gcc could not be found"
    echo "installing gcc"
    brew install gcc
fi
# NumPy build from source
conda run -n $VENV_NAME pip install cython pybind11
# NumPy 1.25 np.test() 모두 통과. 1.26.2 통과하지 못함 (2023.12)
conda run -n $VENV_NAME pip install --no-binary :all: numpy~=1.25.0 --no-cache-dir

echo
echo "Tensorflow 설치"
conda run -n $VENV_NAME pip install tensorflow
conda run -n $VENV_NAME pip install tensorflow-metal
conda run -n $VENV_NAME pip install keras-tuner
conda run -n $VENV_NAME pip install keras
echo
echo "데이터 분석 패키지 설치"
conda run -n $VENV_NAME pip install scikit-learn 
conda run -n $VENV_NAME pip install pandas
conda run -n $VENV_NAME pip install matplotlib
echo
echo "주피터 커널 설치"
conda run -n $VENV_NAME pip install ipykernel
