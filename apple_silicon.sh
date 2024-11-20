#!/bin/bash
PYTHON_VERSION=3.10
NUMPY_VERSION=1.26.0
TENSORFLOW_VERSION=2.17.0
TORCH_VERSION=2.5.0
KERAS_VERSION=3.6.0

# 공통 패키지 목록
PYDATA_PACKAGES=("pandas" "scikit-learn" "matplotlib" "pydot" "ipykernel" "ipywidgets" "sentencepiece")

echo -e '\033[1;36m'
echo ' ██████╗ ██████╗ ██████╗ ███████╗██████╗  █████╗ ███████╗██╗ ██████╗'
echo '██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██║██╔════╝'
echo '██║     ██║   ██║██║  ██║█████╗  ██████╔╝███████║███████╗██║██║     '
echo '██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗██╔══██║╚════██║██║██║     '
echo '╚██████╗╚██████╔╝██████╔╝███████╗██████╔╝██║  ██║███████║██║╚██████╗'
echo ' ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝'
echo -e '\033[0m\033[1;32m코드베이직 (c) 2015-2024 \033[0m'
echo "Apple Silicon 딥러닝 환경 설정 스크립트"

build_and_install_numpy() {
    local env_name=$1   # Name of the conda environment
    local numpy_version=$2 # NumPy version to install
    
    echo "NumPy를 빌드 및 설치합니다 (버전: ${numpy_version})" 
    echo "MacOS Accelerate Framework을 BLAS 백엔드로 사용합니다."
    
    # gcc 확인 및 설치
    if ! command -v gcc &> /dev/null; then
        echo "gcc가 발견되지 않았습니다. conda를 통해 '${env_name}' 환경에 gcc를 설치합니다."
        if ! conda install -y -n "${env_name}" -c conda-forge gcc; then
            echo "오류: gcc 설치에 실패했습니다. Conda 설정을 확인하세요."
            exit 1
        fi
    fi
    
    # 빌드 도구 설치
    echo "빌드 도구(cython, pybind11)를 설치합니다."
    if ! conda run -n "${env_name}" pip install cython pybind11; then
        echo "오류: 빌드 도구 설치에 실패했습니다. 인터넷 연결 및 Conda 설정을 확인하세요."
        exit 1
    fi
    
    # NumPy 소스 빌드 및 설치
    echo "NumPy를 소스에서 빌드하여 설치합니다..."
    if ! conda run -n "${env_name}" --no-capture-output pip install --force-reinstall --no-binary :all: numpy~="${numpy_version}" --no-cache-dir; then
        echo "오류: NumPy 빌드 및 설치에 실패했습니다. 필요한 빌드 도구 및 종속성을 확인하세요."
        exit 1
    fi
    
    echo "NumPy가 성공적으로 빌드 및 설치되었습니다."
}

# 플랫폼 확인 (MacOS와 Apple Silicon만 지원)
OS_TYPE="$(uname -s)"
ARCHITECTURE="$(uname -m)"
if [[ "$OS_TYPE" != "Darwin" || "$ARCHITECTURE" != "arm64" ]]; then
    echo "오류: 이 스크립트는 Apple Silicon (MacOS ARM64)에서만 지원됩니다."
    exit 1
fi
echo "플랫폼: Mac (Apple Silicon)"
echo

# --help 옵션 처리
# --help 옵션 처리
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "사용법: apple_silicon.sh [환경 이름] [옵션]"
    echo
    echo "옵션:"
    echo "  --help         도움말을 표시합니다."
    echo "  --build-numpy  NumPy를 소스에서 빌드 및 설치합니다."
    echo
    echo "환경 이름:"
    echo "  tensorflow     TensorFlow 환경을 생성합니다. (기본값)"
    echo "  pytorch        PyTorch 환경을 생성합니다."
    echo
    echo "기본값:"
    echo "  환경 이름을 지정하지 않으면 'tensorflow' 환경이 생성됩니다."
    exit 0
fi

# 옵션 초기화
VENV_NAME=""
BUILD_NUMPY=false

# 첫 번째 인자와 두 번째 인자 처리
if [[ "$1" == "--build-numpy" || "$2" == "--build-numpy" ]]; then
    BUILD_NUMPY=true
fi

if [[ "$1" != "--build-numpy" && "$1" != "" ]]; then
    VENV_NAME=$1
else
    VENV_NAME="tensorflow"  # 기본값
fi

# 허용된 환경 이름 목록
ALLOWED_ENV_NAMES=("tensorflow" "pytorch")

# 환경 이름 유효성 확인
if [[ ! " ${ALLOWED_ENV_NAMES[@]} " =~ " ${VENV_NAME} " ]]; then
    echo "오류: 환경 이름은 tensorflow 또는 pytorch로 설정해야 합니다."
    exit 2
fi

# Conda 환경 존재 여부 확인
if conda info --envs | awk '{print $1}' | grep -qx "${VENV_NAME}"; then
    echo "오류: '${VENV_NAME}' 환경이 이미 존재합니다."
    exit 1
else
    echo "환경 '${VENV_NAME}'을(를) 생성합니다..."
    if ! conda create -y -n "$VENV_NAME" -c conda-forge python="$PYTHON_VERSION" "${PYDATA_PACKAGES[@]}"; then
        echo "오류: Conda 환경 생성에 실패했습니다. 로그를 확인하세요."
        exit 3
    fi
fi

# TensorFlow 설치
if [[ "$VENV_NAME" == "tensorflow" ]]; then
    echo "TensorFlow 기반 패키지를 설치합니다..."
    conda run -n $VENV_NAME pip install tensorflow~=$TENSORFLOW_VERSION
    echo "TensorFlow Metal 패키지를 설치합니다..."
    conda run -n $VENV_NAME pip install tensorflow-metal

# PyTorch 설치
elif [[ "$VENV_NAME" == "pytorch" ]]; then
    echo "PyTorch 기반 패키지를 설치합니다..."
    conda install -y -n $VENV_NAME -c pytorch pytorch=$TORCH_VERSION torchvision torchaudio
    echo "Keras 패키지를 설치합니다..."
    conda run -n $VENV_NAME pip install keras~=$KERAS_VERSION
fi

# NumPy 빌드 및 설치
if [[ "$BUILD_NUMPY" == true ]]; then
    echo "NumPy 빌드를 시작합니다..."
    build_and_install_numpy "$VENV_NAME" "$NUMPY_VERSION"
fi

echo "설치가 완료되었습니다!"

# 환경 사용 안내 메시지 출력
echo
echo "=================================================="
echo "설치된 Conda 환경 '${VENV_NAME}'을(를) 사용하려면:"
echo "  1. Conda 환경 활성화: conda activate ${VENV_NAME}"
echo "  2. 필요한 작업 수행"
echo "  3. 환경 비활성화: conda deactivate"
echo
echo "참고: 환경을 제거하려면 다음 명령을 실행하세요:"
echo "  conda remove --name ${VENV_NAME} --all"
echo "=================================================="
