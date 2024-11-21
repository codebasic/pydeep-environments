# 딥러닝 환경 설정

Codebasic (c) 2015-2024

다음 문서는 아래 플랫폼별 딥러닝 소프트웨어 라이브러리 설치 절차를 안내합니다.

## 플랫폼

1. Windows (x86-64bit)
1. 유닉스 계열 (Unix-Like)
    1. Mac (Apple Silicon/Intel x86-64bit)
    1. Linux (x86-64bit)

윈도우와 리눅스 경우, 직접 설치보다는 환경 구성이 완료된 [도커 사용](#docker)을 권장합니다.

맥은 [직접 설치](#mac)를 권장합니다.

## [Docker](https://docs.docker.com/get-started/overview/)

Docker는 가상화를 위한 오픈 소스 소프트웨어입니다.

Docker Desktop은 도커 환경 관리를 위한 GUI 인터페이스 소프트웨어입니다. 무료로 설치가 가능하지만 상용 라이선스 소프트웨어입니다. 개인 및 중소 규모 조직은 무료로 사용할 수 있습니다.

정부 기관 및 대기업 환경에서 Docker Desktop 활용 시 라이선스를 검토하시기 바랍니다. 상용 라이선스 소프트웨어 설치와 활용에 대한 우려가 있는 경우, 1) 리눅스에서 도커를 설정하거나, 2) 직접 설치 절차를 진행하기 바랍니다.

### Docker Desktop for Windows

[Docker Desktop for Windows 설치](https://docs.docker.com/desktop/install/windows-install)

#### 요구사항

* Windows 10 이상 64비트 (x86-64)
* [WSL 설치](https://learn.microsoft.com/ko-kr/windows/wsl/install#install-wsl-command)

### Linux (Ubuntu)

[ubuntu_setup.sh](https://github.com/codebasic/pydeep-environments/blob/main/ubuntu_setup.sh) 파일을 참조하여 다음과 같이 도커 환경을 설정합니다.

```sh
sudo bash ubuntu_setup.sh
```

### 도커 컨테이너 실행

최초 실행 시, 도커 이미지([codebasic/pydeep](https://hub.docker.com/r/codebasic/pydeep)) 다운로드가 실행됩니다.

다음 중 실행 환경에 따라 *하나를 선택*하여 실행합니다.

#### GPU 가속 활용

지원하는 NVIDIA 그래픽 카드 장치가 장착되어 있는 경우. 장치의 [드라이버](https://www.nvidia.co.kr/drivers) 설치 및 갱신이 필요할 수 있습니다.

```powershell
docker run --name pydeep-gpu -p 8888:8888 --gpus all -it codebasic/pydeep
```

#### CPU 기반

딥러닝 소프트웨어의 GPU 가속을 활용하지 않거나, 활용할 수 없는 경우

```powershell
docker run --name pydeep -p 8888:8888 -it codebasic/pydeep
```

## 직접 설치 (Native)

### Miniconda 설치

Conda 환경 관리를 위해 Miniconda 설치 파일을 다운로드하고 설치합니다. Miniconda는 Python 환경을 손쉽게 관리할 수 있는 도구입니다.

#### Miniconda Window

[Miniconda Windows Installer](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) 다운로드 및 실행 (설치 마법사)

#### Miniconda Mac

```sh
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

설치 완료 후, 쉘에서 conda 명령을 활용할 수 있도록 설정

```sh
conda init --all
```

이후 절차는 **새 터미널**에서 진행합니다.

### Windows

GPU 가속을 활용하고자 하는 경우, 도커 활용([Docke Desktop for Windows](#docker-desktop-for-windows))을 권장합니다. 하지만 직접 설치를 원할 경우 아래 절차를 따르십시오.

#### 텐서플로우 환경

```powershell
conda create --name tensorflow python=3.10
conda activate tensorflow
```

활성화된 환경에서 [Tensorflow](https://www.tensorflow.org/install) 문서의 설치 절차를 따릅니다.

#### PyTorch 환경

```powershell
conda create --name pytorch python=3.10
conda activate pytorch
```

활성화된 환경에서 [PyTorch](https://pytorch.org/get-started/locally/) 문서의 설치 절차를 따릅니다.

### Mac

애플 실리콘(Apple Silicon)에서는 Apple Metal API로 GPU 가속이 가능합니다. 별도의 드라이버 설치가 필요하지 않습니다. [Apple 개발자 문서: Tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)

[apple_silicon.sh](https://github.com/codebasic/pydeep-environments/blob/main/apple_silicon.sh) 파일을 참조하여 설치를 진행합니다.

```sh
bash ./apple_silicon.sh --help
```

### [선택적] Jupyter

코드 작성 환경 (IDE) Jupyter Lab 설치. [직접 설치](#직접-설치-native)를 진행한 경우를 가정합니다.

도커 환경을 활용하는 경우는 설치와 설정이 완료되어 있습니다.

Jupyter Lab 설치

콘다 환경이 생성되어 있고, 명칭이 "tensorflow"인 경우를 가정합니다. 다른 경우, 환경 변수에서 명칭을 변경한 뒤 사용하기 바랍니다.

```bash
ENV_NAME=tensorflow conda run -n $ENV_NAME pip install jupyterlab
```

파이썬 환경을 주피터 커널로 등록합니다. 주의! 한글 사용자명. 예: C:\Users\성주

```bash
ENV_NAME=tensorflow conda run -n $ENV_NAME python -m ipykernel install --user --name $ENV_NAME --display-name $ENV_NAME
```
