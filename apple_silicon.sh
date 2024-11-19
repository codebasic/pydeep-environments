#!/bin/bash
# 환경 이름을 인자로 받기
VENV_NAME=$1  # 첫 번째 인자를 환경 이름으로 사용
# 인자가 없는 경우 기본값 설정
if [ -z "$VENV_NAME" ]; then
    VENV_NAME="tensorflow"  # 기본값: tensorflow
fi

PYTHON_VERSION=3.10
NUMPY_VERSION=1.25.0
TENSORFLOW_VERSION=2.17.0
TORCH_VERSION=2.5.0
KERAS_VERSION=3.6.0

# 공통 패키지 목록
PYDATA_PACKAGES="pandas scikit-learn matplotlib pydot ipykernel ipywidgets sentencepiece"

echo -e '\033[1;36m'
echo ' ██████╗ ██████╗ ██████╗ ███████╗██████╗  █████╗ ███████╗██╗ ██████╗'
echo '██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██║██╔════╝'
echo '██║     ██║   ██║██║  ██║█████╗  ██████╔╝███████║███████╗██║██║     '
echo '██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗██╔══██║╚════██║██║██║     '
echo '╚██████╗╚██████╔╝██████╔╝███████╗██████╔╝██║  ██║███████║██║╚██████╗'
echo ' ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝'
echo -e '\033[0m\033[1;32m코드베이직 (c) 2015-2024 \033[0m'
echo "Apple Silicon 딥러닝 환경 설정 스크립트"

# Function to build and install NumPy from source
build_and_install_numpy() {
    local env_name=$1   # Name of the conda environment
    local numpy_version=$2 # NumPy version to install
    
    echo "NumPy를 빌드 및 설치합니다 (버전: ${numpy_version})" 
    echo "MacOS Accelerate Framework을 BLAS 백엔드로 사용합니다."
    if ! command -v gcc &> /dev/null; then
        echo "gcc가 발견되지 않았습니다. conda를 통해 '${env_name}' 환경에 gcc를 설치합니다."
        conda install -y -n ${env_name} -c conda-forge gcc
    fi
    # Install necessary build tools
    conda run -n "$env_name" pip install cython pybind11    
    # Install NumPy from source with the specified version
    conda run -n "$env_name" pip install --no-binary :all: "numpy~=${numpy_version}" --no-cache-dir
}

# 플랫폼 확인 (MacOS와 Apple Silicon만 지원)
OS_TYPE="$(uname -s)"
ARCHITECTURE="$(uname -m)"
if [[ "$OS_TYPE" != "Darwin" || "$ARCHITECTURE" != "arm64" ]]; then
    echo "오류: 이 스크립트는 Apple Silicon (MacOS ARM64)에서만 지원됩니다."
    exit 1
fi
echo "플랫폼: Mac (Apple Silicon)"

# 환경 이름이 "tensorflow" 또는 "pytorch"인지 확인
if [[ "$VENV_NAME" != "tensorflow" && "$VENV_NAME" != "pytorch" ]]; then
    echo "오류: 환경 이름은 'tensorflow' 또는 'pytorch' 중 하나여야 합니다."
    exit 1
fi

# Conda 환경 목록에서 환경 이름이 존재하는지 확인
if conda env list | grep -q "^\s*${VENV_NAME}\s"; then
    echo "환경 '${VENV_NAME}'이(가) 이미 존재합니다."; exit 1
else
    echo "환경 '${VENV_NAME}'이(가) 존재하지 않습니다. 새 환경을 생성합니다..."
    conda create -y -n $VENV_NAME -c conda-forge python=$PYTHON_VERSION \
        pandas scikit-learn matplotlib pydot ipykernel ipywidgets sentencepiece
fi

# TensorFlow 설치
if [ "$VENV_NAME" == "tensorflow" ]; then
    echo "Apple Silicon에서 TensorFlow와 관련 패키지를 설치합니다..."
    conda run -n $VENV_NAME pip install tensorflow~=$TENSORFLOW_VERSION
    echo "TensorFlow Metal 패키지를 설치합니다..."
    conda run -n $VENV_NAME pip install tensorflow-metal
    # # NumPy를 빌드 및 설치
    # build_and_install_numpy "$VENV_NAME" "$NUMPY_VERSION"

# PyTorch 설치
elif [ "$VENV_NAME" == "pytorch" ]; then
    echo "PyTorch 패키지를 설치합니다..."
    conda install -y -n $VENV_NAME -c pytorch pytorch=$TORCH_VERSION torchvision torchaudio
    echo "Keras 패키지를 설치합니다..."
    conda run -n $VENV_NAME pip install keras~=$KERAS_VERSION
    # # NumPy를 빌드 및 설치
    # build_and_install_numpy "$VENV_NAME" "$NUMPY_VERSION"
fi

echo "설치가 완료되었습니다!"
