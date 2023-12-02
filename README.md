Codebasic (c) 2023

다음 문서는 아래 플랫폼별 딥러닝 소프트웨어 라이브러리 설치 절차를 안내합니다.

# 플랫폼

1. Windows (x86-64bit)
1. 유닉스 계열 (Unix-Like)
    1. Mac (Apple Silicon/Intel x86-64bit)
    1. Linux (x86-64bit)

윈도우와 리눅스 경우, 직접 설치보다는 환경 구성이 완료된 도커 사용을 권장합니다. 

맥은 직접 설치를 권장합니다. 애플 실리콘 맥에서는 x86 기반 도커 이미지가 정상적으로 동작하지 않습니다.

# [Docker](https://docs.docker.com/get-started/overview/)
## Windows
### Docker Desktop for Windows

Docker Desktop은 무료로 설치가 가능하지만 상용 라이선스 소프트웨어입니다. 개인 및 중소 규모 조직은 무료로 사용할 수 있습니다. 

정부 기관 및 대기업 환경에서 활용 시 라이선스를 검토하시기 바랍니다. 상용 라이선스 소프트웨어 설치와 활용에 대한 우려가 있는 경우, 1) 리눅스에서 도커를 설정하거나, 2) 직접 설치 절차를 진행하기 바랍니다.

https://docs.docker.com/desktop/install/windows-install

#### 요구사항

* Windows 10 이상 64비트 (x86-64)
* [WSL 설치](https://learn.microsoft.com/ko-kr/windows/wsl/install#install-wsl-command)

## Linux (Ubuntu)

[ubuntu_setup.sh](ubuntu_setup.sh) 파일 필요.

```bash
sudo source ubuntu_setup.sh
```

## 도커 컨테이너 실행

최초 실행 시, 약 3 GB 용량의 도커 이미지([codebasic/pydeep](https://hub.docker.com/r/codebasic/pydeep)) 다운로드가 실행됩니다.

다음 중 실행 환경에 따라 *하나를 선택*하여 실행합니다.

### GPU 가속 활용

지원하는 NVIDIA 그래픽 카드 장치가 장착되어 있는 경우 ([최신 그래픽 드라이버](https://www.nvidia.co.kr/Download/index.aspx?lang=kr) 설치가 필요할 수 있습니다.)

```powershell
docker run --name pydeep-gpu -p 8888:8888 --gpus all -it codebasic/pydeep
```

### CPU 기반

딥러닝 소프트웨어의 GPU 가속을 활용하지 않거나, 활용할 수 없는 경우. 

```powershell
docker run --name pydeep -p 8888:8888 -it codebasic/pydeep
```

# 직접 설치 (Native)

제시된 절차는 오픈 소스 라이선스 소프트웨어만을 활용하고 있습니다.

## conda

Conda는 패키지 관리 프로그램입니다. 소프트웨어 버전과 의존성 관리에 활용합니다.

### Windows

[Miniconda Windows](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) 다운로드 및 설치

### Mac

아래 절차는 [Homebrew](https://brew.sh/index_ko) 소프트웨어를 가정합니다.

```bash
brew install miniconda
conda init "$(basename "${SHELL}")"
```

### Linux

[Miniforge](https://github.com/conda-forge/miniforge) 설치
```bash
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
./Miniforge3.sh
conda init "$(basename "${SHELL}")"
```

## 딥러닝 소프트웨어

다음 명령을 실행하여 설치합니다.

environment.yml 파일은 각 플랫폼별 환경 설정 파일을 참조합니다.

1. [x86_gpu.yml](x86_gpu.yml)
1. [x86_cpu.yml](x86_cpu.yml)


### Windows

```powershell
conda env create -f environment.yml
```

### Mac

[apple_silicon.sh](apple_silicon.sh) 파일 필요.

```zsh
source ./apple_silicon.sh
```

### Linux

```bash
conda env create -f environment.yml
```

## GPU 가속

Winodws 및 Linux 플랫폼에서 [지원하는 NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) 기반 그래픽 카드 필요.

### Windows

TensorFlow (2.10 이하) 동작을 위해 [Microsoft Visual C++ 재배포 가능 패키지 설치 필요 (64비트)](https://aka.ms/vs/17/release/vc_redist.x64.exe)

Tensorflow 2.11+ 에서는 직접(native) 설치를 통한 GPU 가속을 지원하지 않음. WSL2 활용 필요.

### Mac

Apple Silicon은 추가 설정 없이 GPU 가속 가능. Intel 기반 맥은 GPU 가속을 지원하지 않음.

[Apple 개발자 문서: Tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)

### Linux

NVIDIA CUDA 라이브러리 탐색 경로 설정
```bash
conda activate pydeep
source ./set_libs.sh
conda deactivate && conda activate pydeep
```

[set_libs.sh](set_libs.sh) 참조.

##  [선택적] Jupyter

코드 작성 환경 (IDE) Jupyter Lab 설치.

주의! 한글 사용자명. 예: C:\Users\성주

```bash
conda install -n pydeep -c conda-forge jupyterlab
conda run -n pydeep python -m ipykernel install --user --name pydeep --display-name "pydeep"
```