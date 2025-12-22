# 딥러닝 환경 설정

Codebasic (c) 2015-2025

다음 문서는 아래 플랫폼별 딥러닝 소프트웨어 라이브러리 설치 절차를 안내합니다.

## 플랫폼

1. Windows (x86-64bit)
1. 유닉스 계열 (Unix-Like)
    1. Mac (Apple Silicon/Intel x86-64bit)
    1. Linux (x86-64bit)

윈도우와 리눅스 경우, 직접 설치보다는 환경 구성이 완료된 도커 사용을 권장합니다.

맥은 직접 설치를 권장합니다.

## [Docker](https://docs.docker.com/get-started/overview/)

Docker는 가상화를 위한 오픈 소스 소프트웨어입니다.

### Windows

윈도우에서는 도커 엔진과 관리 프로그램을 한번에 설치하는 [Docker Desktop](https://docs.docker.com/desktop/) 활용을 권장합니다. Docker Desktop은 도커 환경 관리를 위한 GUI 인터페이스 소프트웨어입니다. 개인/중소 규모 조직은 무료로 사용할 수 있습니다.

[도커 엔진](https://docs.docker.com/engine/)은 무료/오픈 소스이지만, GUI 기반 관리 편의를 제공하는 Docker Desktop은 상용 라이선스 소프트웨어입니다. 정부 기관 및 대기업 환경에서 Docker Desktop 활용 시 라이선스를 검토하시기 바랍니다. 상용 라이선스 소프트웨어 설치와 활용에 대한 우려가 있는 경우, 도커 엔진만 설치하거나, 오픈 소스 기반 도커 관리 도구를 활용할 수 있습니다.

[Docker Desktop for Windows 설치](https://docs.docker.com/desktop/install/windows-install)

#### 요구사항

* Windows 10 이상 64비트 (x86-64)
* [WSL 설치](https://learn.microsoft.com/ko-kr/windows/wsl/install#install-wsl-command)

#### GPU 가속 활용

지원하는 NVIDIA 그래픽 카드 장치가 장착되어 있는 경우. 장치의 [드라이버](https://www.nvidia.co.kr/drivers) 설치 및 갱신이 필요할 수 있습니다.

[NVIDIA GPU 가속 확인 (GPU support in Docker Desktop for Windows)](https://docs.docker.com/desktop/features/gpu)

### Linux

리눅스는 각 배포판별 도커 설치 절차를 참조하시기 바랍니다.

[Docker Engine 설치 안내 (Install Docker Engine)](https://docs.docker.com/engine/install/)

### 도커 컨테이너 실행

최초 실행 시, 도커 이미지([codebasic/pydeep](https://hub.docker.com/r/codebasic/pydeep)) 다운로드가 실행됩니다.

```sh
docker run --name pydeep --gpus=all --shm-size=2g -p 8888:8888 -d codebasic/pydeep
```

주요 설정

* [GPU 접근](https://docs.docker.com/engine/containers/resource_constraints/#gpu)
* [포트 연결](https://docs.docker.com/engine/network/port-publishing/)  
    주피터 서버 접근을 위해 호스트 포트를 컨테이너 내부 포트에 연결합니다.
* 공유 메모리  
    도커 컨테이너 공유 메모리 (shared memory) 크기 기본값(64MB)이 작아서 멀티 프로세스가 실패할 수 있습니다. ([도커 컨테이너 자원 제한](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources))  
    다음 중 하나의 설정을 권장합니다.  
    1. 공유 메모리 크기 설정 (`shm-size`)
    2. [호스트 공유 메모리 활용](https://docs.docker.com/reference/cli/docker/container/run/#ipc)

#### 활용 예시

호스트 디렉터리를 컨테이너에 [바인드 마운트(bind mount)](https://docs.docker.com/engine/storage/bind-mounts/)하여 실행합니다.

POSIX Shell (bash/zsh 등)

```bash
docker run --name pydeep --gpus=all --shm-size=2g -p 8888:8888 \
    -v "$(pwd)/pydeep":/workspace/pydeep \
    -d codebasic/pydeep
```

Powershell

파워쉘은 기존 쉘과 문법 차이가 있습니다.

* 명령줄에서 탈출문자가 백틱(`)으로, 유닉스 계열의 역슬래시(```\```)와 구분됩니다.
* 윈도우 경로 구분자는 역슬래시(`\`)로 유닉스 계열의 경로 구분자인 슬래시(`/`)와 차이가 있습니다.  
  바인드 마운트 시, 윈도우 호스트 경로에서 사용하는 경로 구분자와 리눅스 컨테이너 경로 작성 시 유의해야 합니다.  
* 현재 경로값 획득 시, 경로 치환 방식이 다릅니다: POSIX Shell은 `$(pwd)`처럼 명령 치환을 쓰고, PowerShell은 현재 경로값을 담은 변수 `${pwd}` 를 사용합니다.

```powershell
docker run --name pydeep --gpus=all --shm-size=2g -p 8888:8888 `
    -v "${pwd}\pydeep":/workspace/pydeep `
    -d codebasic/pydeep
```

#### GPU 확인

컨테이너 내부 쉘에서 다음 명령으로 GPU 인식 여부를 확인합니다.

참고: macOS에서는 Docker 컨테이너 내 NVIDIA GPU 가속을 사용할 수 없습니다. Mac은 호스트 직접 설치로 MPS/Metal 가속을 활용하는 것을 권장합니다.

```sh
nvidia-smi
```

#### 주피터 서버

컨테이너에서 주피터(jupyter) 서버가 실행 중인 경우, 호스트 웹브라우저에서 다음 주소로 접속합니다.

`http://localhost:8888`

토큰 값이 필요한 경우, 도커 컨테이너 쉘에서 다음 명령으로 토큰 값을 확인합니다.

```sh
docker exec pydeep jupyter server list
```

컨테이너 내부의 URL은 호스트의 `localhost`로 접속해야 합니다. 다음 명령으로 호스트 주소로 치환할 수 있습니다.

* 파워쉘(Windows PowerShell)

```powershell
docker exec pydeep jupyter server list | ForEach-Object { $_ -replace 'http://[^:]+', 'http://localhost' }
```

* POSIX 쉘 (bash/zsh 등)
  
```sh
docker exec pydeep jupyter server list | sed -E 's#http://[^:]+#http://localhost#'
```

## 직접 설치

### Miniforge 설치

[conda 설치 (miniforge)](https://conda-forge.org/download/)

### 공통 패키지 설치

```sh
conda create --name pyml python=3.10
conda install --name pyml scikit-learn pandas matplotlib ipykernel ipywidgets
```

### 딥러닝 프레임워크

#### PyTorch

* 설치 안내: [PyTorch](https://pytorch.org/get-started/locally/) 공식 문서를 참조해 OS/하드웨어에 맞게 설치합니다.
* 환경 생성: 공통 환경(`pyml`)을 복제해 PyTorch 전용 환경을 만듭니다.

```sh
conda create --name pytorch --clone pyml
```

#### TensorFlow

* 설치 안내: [Tensorflow](https://www.tensorflow.org/install) 공식 문서를 참조해 OS/하드웨어에 맞게 설치합니다.
* 환경 생성: 공통 환경(`pyml`)을 복제해 TensorFlow 전용 환경을 만듭니다.

```sh
conda create --name tensorflow --clone pyml
```

### [선택적] Jupyter

코드 작성 환경 (IDE) Jupyter Lab 설치.

명령줄 도구는 [Astral uv](https://docs.astral.sh/uv/)로 설치를 권장합니다.

```sh
# Astral UV를 통해 최신 버전의 Jupyter Lab을 실행합니다.
uvx --from jupyterlab jupyter-lab
```

주피터 실행

```sh
jupyter-lab
```

#### 주피터 커널

주피터에서 파이썬 환경을 사용하려면 각 환경을 주피터 커널(kernel)로 등록해야 합니다. 각 환경을 활성화 후, 다음 명령을 실행합니다.

예: PyTorch 환경 등록

```sh
conda activate pytorch
python -m ipykernel install --user --name pytorch --display-name "Pytorch 2"
```
