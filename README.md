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

## Docker

[도커(Docker)](https://docs.docker.com/get-started/overview/)는 가상화를 위한 오픈 소스 소프트웨어입니다.

### Windows

윈도우에서는 도커 엔진과 관리 프로그램을 한번에 설치하는 [Docker Desktop](https://docs.docker.com/desktop/) 활용을 권장합니다. Docker Desktop은 도커 환경 관리를 위한 GUI 인터페이스 소프트웨어입니다. 개인/중소 규모 조직은 무료로 사용할 수 있습니다.

[도커 엔진](https://docs.docker.com/engine/)은 무료/오픈 소스이지만, GUI 기반 관리 편의를 제공하는 Docker Desktop은 상용 라이선스 소프트웨어입니다. 정부 기관 및 대기업 환경에서 Docker Desktop 활용 시 라이선스를 검토하시기 바랍니다. 상용 라이선스 소프트웨어 설치와 활용에 대한 우려가 있는 경우, 도커 엔진만 설치하거나, 오픈 소스 기반 도커 관리 도구를 활용할 수 있습니다.

[Docker Desktop for Windows 설치](https://docs.docker.com/desktop/install/windows-install)

#### 요구사항

* Windows 10 64비트 22H2, Windows 11 64비트 23H2 이상 (x86-64)
* [WSL 설치](https://learn.microsoft.com/ko-kr/windows/wsl/install#install-wsl-command)

    ```powershell
    wsl --update
    ```

<!-- ![WSL 업데이트 화면](wsl-update.png) -->

#### GPU 가속 활용

지원하는 NVIDIA 그래픽 카드 장치가 장착되어 있는 경우. 장치의 [드라이버](https://www.nvidia.co.kr/drivers) 설치 및 갱신이 필요할 수 있습니다.

[NVIDIA GPU 가속 확인 (GPU support in Docker Desktop for Windows)](https://docs.docker.com/desktop/features/gpu)

### Linux

리눅스는 각 배포판별 도커 설치 절차를 참조하시기 바랍니다.

[Docker Engine 설치 안내 (Install Docker Engine)](https://docs.docker.com/engine/install/)

### 도커 컨테이너 실행

공용 이미지 `codebasic/pydeep:26.10`은 Linux x86-64 (`linux/amd64`)용입니다.
JupyterLab 4.6.4, Python 3.12.16 실습 커널, 한국어 팩과 Codebasic 로고가 포함되어 있습니다.
Matplotlib은 홈 설정의 Noto Sans CJK KR 글꼴과 DejaVu Sans 수식 글꼴을 사용하며,
유니코드 마이너스 표시를 지원합니다. Conda와 uv 구성은 유지했고, ML/DL 라이브러리 전반을 일괄 업그레이드한 이미지는 아닙니다.

먼저 이미지를 받습니다. 재현 가능한 실습에는 `latest` 대신 `26.10` 태그를 사용합니다.

```sh
docker pull codebasic/pydeep:26.10
```

실습 파일을 보관할 폴더로 이동한 뒤 실행합니다. 현재 폴더가 컨테이너의 `/workspace`에 연결됩니다.

PowerShell:

```powershell
docker run --name pydeep --gpus=all --shm-size=2g -p 127.0.0.1:8888:8888 -v "${pwd}:/workspace" -d codebasic/pydeep:26.10
```

bash/zsh:

```bash
docker run --name pydeep --gpus=all --shm-size=2g -p 127.0.0.1:8888:8888 -v "$(pwd):/workspace" -d codebasic/pydeep:26.10
```

NVIDIA GPU를 사용하지 않는 컴퓨터에서는 `--gpus=all`을 제외합니다.
공유 메모리는 `--shm-size=2g`로 확보합니다. Linux에서 필요하면 기존 방식인 `--ipc=host`로 대체할 수 있습니다.

실행 후 웹브라우저에서 **[http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab)** 을 엽니다.
기본 명령 `jupyter lab`이 컨테이너 내부 `0.0.0.0:8888`에서 자동 실행되므로 별도 설치나 서버 시작 명령이 필요하지 않습니다.
토큰과 비밀번호 없이 접속합니다. Windows에서 `localhost`의 IPv6 경로가 시간 초과될 수 있어 `127.0.0.1`을 권장합니다.

> 무인증 서버의 공개 범위는 Docker 포트 게시 설정에 달려 있습니다. 위 예시는 호스트의 IPv4 루프백에만 연결하며, `-p 8888:8888`은 외부 접근을 허용할 수 있습니다.

노트북에서는 Python 3.12.16 실습 커널을 선택합니다. 실습 파일은 `/workspace` 아래에 저장해야 호스트 폴더에 남습니다.

#### 다시 실행·문제 확인

이미 만든 컨테이너는 새로 만들지 않고 시작합니다.

```sh
docker start pydeep
```

기존 컨테이너에는 새 이미지를 받아도 자동 적용되지 않습니다. 교체 전 `/workspace` 바깥에 저장한 파일도 따로 보관하고,
기존 컨테이너를 중지·제거한 뒤 같은 작업 폴더로 위 실행 명령을 사용합니다.

접속이 안 되면 컨테이너 상태와 로그를 확인합니다.

```sh
docker ps -a --filter name=pydeep
docker logs pydeep
```

#### 선택: 고급 설정·확인

저장소의 `compose.yaml`을 사용하는 경우 실습 폴더에 파일을 놓고 `docker compose up -d`로 시작할 수 있습니다.
현재 폴더를 `/workspace`에 연결하며, 공유 메모리는 기존 예시대로 4GB를 사용합니다. GPU가 없으면 `gpus: all`을 제외합니다.
홈에 포함된 Matplotlib 설정을 그대로 사용하므로 별도 `matplotlibrc` 마운트는 필요하지 않습니다.

* GPU 확인: 호스트에서 `docker exec pydeep nvidia-smi`를 실행합니다.
* 이미지 파일로 전달받은 경우: `docker load --input pydeep.tar` 후 태그를 확인합니다.
* 포트·공유 메모리·마운트 옵션: [포트 게시](https://docs.docker.com/engine/network/port-publishing/), [바인드 마운트](https://docs.docker.com/engine/storage/bind-mounts/), [공유 메모리](https://docs.docker.com/reference/cli/docker/container/run/#ipc)를 참조합니다.
* `26.10`과 `latest`의 확인된 digest는 `sha256:6a364b060331a42b2c54d3441aa228031773ea504b5c483487105dd7902ec47d`입니다. `latest`는 이후 변경될 수 있습니다.

## 코드 에이전트

1. Docker가 실행 중인 **호스트 터미널**에서 실행 중인 컨테이너에 진입합니다.

   ```sh
   docker exec -it pydeep bash
   ```

2. 진입한 **Linux 컨테이너 안에서** 원하는 코드 에이전트를 설치합니다. 아래 공식 문서의 **Linux 설치 절차**를 따르세요. 호스트가 Mac이나 Windows여도 컨테이너에서는 Linux용 안내를 사용합니다.

   * [Claude Code 설치 안내](https://code.claude.com/docs/en/setup#install-claude-code)
   * [OpenAI Codex CLI 설치 안내](https://learn.chatgpt.com/docs/codex/cli)

3. 각 공식 문서에 따라 인증한 뒤 **같은 컨테이너 안에서** 코드 에이전트를 실행합니다.

설치 명령과 인증 방법은 변경될 수 있으므로 각 공식 문서의 최신 안내를 참조하세요.

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

##### GPU 가속 확인

PyTorch가 정상 설치되고 GPU 가속에 접근 가능한지 확인합니다. 환경을 활성화한 후 다음 명령을 실행합니다.

###### CUDA (Linux/Windows)

```sh
conda activate pytorch
python -c "import torch; print('PyTorch Version:', torch.__version__); assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA Device:', torch.cuda.get_device_name(0))"
```

명령이 실행되면 PyTorch 버전을 먼저 출력하고, CUDA를 확인합니다. CUDA가 없으면 `AssertionError`가 발생합니다.

###### MPS (Mac)

Mac에서는 Metal Performance Shaders(MPS)를 사용합니다.

```sh
conda activate pytorch
python -c "import torch; print('PyTorch Version:', torch.__version__); assert torch.backends.mps.is_available(), 'MPS not available'; print('MPS Device:', 'Apple Metal Performance Shaders')"
```

MPS가 사용 가능하면 명령이 실행되고, 그렇지 않으면 `AssertionError`가 발생합니다.

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
