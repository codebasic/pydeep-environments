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

### 1. Miniconda 설치

Conda 환경 관리를 위해 Miniconda 설치 파일을 다운로드하고 설치합니다. Miniconda는 Python 환경을 손쉽게 관리할 수 있는 무료 소프트웨어입니다.

#### Miniconda Window

[Miniconda Windows Installer](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) 다운로드 및 실행 (설치 마법사)

#### Miniconda Mac/Linux

[Miniconda 빠른 명령줄 설치](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) 문서에서 절차와 명령을 참조하여 실행합니다.

### 2. 설치 스크립트 실행

아래 절차를 따라 진행하세요.

#### 실행 환경 준비

* **Windows**: Anaconda Powershell Prompt 실행
* **Mac/Linux**: 터미널 실행

#### 설치 절차

[codebasic (PyPI)](https://pypi.org/project/codebasic/) 모듈 설치

```sh
pip install codebasic
```

다음 명령을 실행하여 설치 스크립트의 안내를 참조하여 진행합니다.

```sh
python -m codebasic --help
```
