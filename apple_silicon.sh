#!/bin/bash
echo -e '\033[1;36m'
echo ' ██████╗ ██████╗ ██████╗ ███████╗██████╗  █████╗ ███████╗██╗ ██████╗'
echo '██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██║██╔════╝'
echo '██║     ██║   ██║██║  ██║█████╗  ██████╔╝███████║███████╗██║██║     '
echo '██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗██╔══██║╚════██║██║██║     '
echo '╚██████╗╚██████╔╝██████╔╝███████╗██████╔╝██║  ██║███████║██║╚██████╗'
echo ' ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝'
echo -e '\033[0m\033[1;32m코드베이직 (c) 2015-2024 \033[0m'
echo "Apple Silicon 딥러닝 환경 설정 스크립트"

# 공통 패키지 목록
PYDATA_PACKAGES="pandas scikit-learn matplotlib pydot ipykernel ipywidgets sentencepiece"
HF_PACKAGES="transformers datasets"

# 환경 이름을 인자로 받기
VENV_NAME=$1  # 첫 번째 인자를 환경 이름으로 사용

PYTHON_VERSION=3.10
TENSORFLOW_VERSION=2.17.0
TORCH_VERSION=2.2.0

# 환경 이름이 "tensorflow" 또는 "pytorch"인지 확인
if [[ "$VENV_NAME" != "tensorflow" && "$VENV_NAME" != "pytorch" ]]; then
    echo "오류: 환경 이름은 'tensorflow' 또는 'pytorch' 중 하나여야 합니다."
    exit 1
fi

# Conda 환경 목록에서 환경 이름이 존재하는지 확인
if conda env list | grep -q "^\s*${VENV_NAME}\s"; then
    echo "환경 '${VENV_NAME}'이(가) 이미 존재합니다."
else
    echo "환경 '${VENV_NAME}'이(가) 존재하지 않습니다. 새 환경을 생성합니다..."
    conda create -y -n $VENV_NAME -c conda-forge python=$PYTHON_VERSION
fi

echo "환경 '${VENV_NAME}'에 데이터 분석 및 Hugging Face 패키지를 설치합니다..."
conda install -y -n ${VENV_NAME} --override-channels -c conda-forge -c huggingface ${PYDATA_PACKAGES} ${HF_PACKAGES}

if [ "$VENV_NAME" == "tensorflow" ]; then
    if ! command -v gcc &> /dev/null; then
        echo "gcc가 발견되지 않았습니다. conda를 통해 '${VENV_NAME}' 환경에 gcc를 설치합니다."
        conda install -y -n ${VENV_NAME} --override-channels -c conda-forge gcc
    fi
    echo "NumPy를 설치합니다 (호환성을 위해 버전 1.25 사용)..."
    conda run -n $VENV_NAME pip install cython pybind11
    conda run -n $VENV_NAME pip install --no-binary :all: numpy~=1.25.0 --no-cache-dir
    echo "TensorFlow와 Metal 지원을 설치합니다..."
    conda run -n $VENV_NAME pip install tensorflow~=$TENSORFLOW_VERSION
    conda run -n $VENV_NAME pip install tensorflow-metal
elif [ "$VENV_NAME" == "pytorch" ]; then
    echo "PyTorch 패키지를 설치합니다..."
    conda install -y -n $VENV_NAME --override-channels -c conda-forge -c pytorch pytorch~=$TORCH_VERSION torchvision torchaudio
fi

echo "설치가 완료되었습니다! 'conda activate $VENV_NAME' 명령어로 환경을 활성화할 수 있습니다."