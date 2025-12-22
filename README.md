# 딥러닝 환경 설정

Codebasic (c) 2015-2025

다음 문서는 아래 플랫폼별 딥러닝 소프트웨어 라이브러리 설치 절차를 안내합니다.

## 플랫폼

1. Windows (x86-64bit)
1. 유닉스 계열 (Unix-Like)
    1. Mac (Apple Silicon/Intel x86-64bit)
    1. Linux (x86-64bit)

윈도우와 리눅스 경우, 직접 설치보다는 환경 구성이 완료된 [도커 사용](#docker)을 권장합니다.

맥은 직접 설치를 권장합니다.

## [Docker](https://docs.docker.com/get-started/overview/)

Docker는 가상화를 위한 오픈 소스 소프트웨어입니다.

Docker Desktop은 도커 환경 관리를 위한 GUI 인터페이스 소프트웨어입니다. 무료로 설치가 가능하지만 상용 라이선스 소프트웨어입니다. 개인 및 중소 규모 조직은 무료로 사용할 수 있습니다.

정부 기관 및 대기업 환경에서 Docker Desktop 활용 시 라이선스를 검토하시기 바랍니다. 상용 라이선스 소프트웨어 설치와 활용에 대한 우려가 있는 경우, 1) 리눅스에서 도커를 설정하거나, 2) 직접 설치 절차를 진행하기 바랍니다.

### Docker Desktop for Windows

[Docker Desktop for Windows 설치](https://docs.docker.com/desktop/install/windows-install)

#### 요구사항

* Windows 10 이상 64비트 (x86-64)
* [WSL 설치](https://learn.microsoft.com/ko-kr/windows/wsl/install#install-wsl-command)

#### GPU 가속 활용

지원하는 NVIDIA 그래픽 카드 장치가 장착되어 있는 경우. 장치의 [드라이버](https://www.nvidia.co.kr/drivers) 설치 및 갱신이 필요할 수 있습니다.

[NVIDIA GPU 가속 확인 (GPU support in Docker Desktop for Windows)](https://docs.docker.com/desktop/features/gpu)

### Linux (Ubuntu)

[ubuntu_setup.sh](https://github.com/codebasic/pydeep-environments/blob/main/ubuntu_setup.sh) 파일을 참조하여 다음과 같이 도커 환경을 설정합니다.

```sh
sudo bash ubuntu_setup.sh
```

### 도커 컨테이너 실행

최초 실행 시, 도커 이미지([codebasic/pydeep](https://hub.docker.com/r/codebasic/pydeep)) 다운로드가 실행됩니다.

```sh
docker run --name pydeep --gpus=all --shm-size=2g -p 8888:8888 -it codebasic/pydeep
```

주요 설정

* [GPU 접근](https://docs.docker.com/engine/containers/resource_constraints/#gpu)
* 공유 메모리
    도커 컨테이너를 사용하는 경우, 공유 메모리 (shm; shared memory) 크기의 기본값(64MB)이 작아서 멀티 프로세스가 실패할 수 있습니다.
    다음 중 하나의 설정을 권장합니다.
    ([도커 컨테이너 자원 제한](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources))
    1. 공유 메모리 크기 설정 (`shm-size`)
    2. [호스트 공유 메모리 활용](https://docs.docker.com/reference/cli/docker/container/run/#ipc)

* [포트 연결](https://docs.docker.com/engine/network/port-publishing/)

    주피터 서버 접근을 위해 호스트 포트를 컨테이너 내부 포트에 연결합니다.

#### 데이터/노트북 볼륨 마운트 예시

호스트 디렉터리를 컨테이너에 마운트해 노트북/데이터를 영속화합니다.

```sh
docker run --name pydeep \
    --gpus=all --shm-size=2g \
    -p 8888:8888 \
    -v "$PWD/notebooks":/workspace/notebooks \
    -v "$PWD/data":/workspace/data \
    -it codebasic/pydeep
```

참고: macOS에서는 Docker 컨테이너 내 NVIDIA GPU 가속을 사용할 수 없습니다. Mac은 호스트 직접 설치로 MPS/Metal 가속을 활용하는 것을 권장합니다.

#### 컨테이너에서 GPU 확인 (Linux/WSL)

컨테이너 내부 쉘에서 다음 명령으로 GPU 인식 여부를 확인합니다.

```sh
nvidia-smi
```

## 직접 설치

### Miniconda 설치

아래 문서에서 각 운영체제별 설치 절차 참조.

[conda 설치 (miniforge)](https://conda-forge.org/download/)

### 공통 패키지 설치

```sh
conda create --name pyml python=3.10
conda install --name pyml scikit-learn pandas matplotlib ipykernel
```

### 프레임워크별 직접 설치

#### PyTorch 환경

* 설치 안내: [PyTorch](https://pytorch.org/get-started/locally/) 공식 문서를 참조해 OS/하드웨어에 맞게 설치합니다.
* 환경 생성: 공통 환경(`pyml`)을 복제해 PyTorch 전용 환경을 만듭니다.

```sh
conda create --name pytorch --clone pyml
```

커널 등록: 주피터에서 선택할 수 있도록 커널을 등록합니다.

```sh
conda run -n pytorch python -m ipykernel install --user --name pytorch --display-name "PyTorch 2"
```

GPU 가속 검증

Linux/Windows (CUDA)

```sh
conda run -n pytorch python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

macOS (MPS)

```sh
conda run -n pytorch python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### TensorFlow 환경

* 설치 안내: [Tensorflow](https://www.tensorflow.org/install?hl=ko) 공식 문서를 참조해 OS/하드웨어에 맞게 설치합니다.
* 환경 생성: 공통 환경(`pyml`)을 복제해 TensorFlow 전용 환경을 만듭니다.

```sh
conda create --name tensorflow --clone pyml
```

커널 등록: 주피터에서 선택할 수 있도록 커널을 등록합니다.

```sh
conda run -n tensorflow python -m ipykernel install --user --name tensorflow --display-name "Tensorflow 2"
```

GPU 가속 검증

Linux/Windows (CUDA)

```sh
conda run -n tensorflow python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPUs: {len(tf.config.list_physical_devices('GPU'))}')"
```

macOS (Metal)

```sh
conda run -n tensorflow python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPUs: {len(tf.config.list_physical_devices('GPU'))}')"
```
