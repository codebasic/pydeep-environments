# 딥러닝 환경 설정

Codebasic (c) 2015-2026

도커 또는 직접 설치로 실습 환경을 구성할 수 있습니다. Windows와 Linux에서는 도커를, Mac에서는 직접 설치를 권장합니다.

## 도커

도커는 애플리케이션 실행에 필요한 소프트웨어와 라이브러리를 컨테이너 이미지로 구성해, 여러 컴퓨터에서 일관된 환경을 제공하는 데 유용합니다. 강의·실습 환경을 이미지로 미리 준비하면 각 컴퓨터의 설치와 설정을 단순화할 수 있습니다.

실행에 필요한 데이터·모델·추가 의존성은 환경에 맞게 준비하고, 사용할 컴퓨터에서 실습이 정상적으로 실행되는지 확인합니다.

### Windows

[Docker Desktop for Windows 설치 안내](https://docs.docker.com/desktop/install/windows-install)에 따라 WSL 2 기반으로 설치하고 Docker Desktop을 실행합니다.
요구사항과 라이선스 조건은 공식 문서를 확인합니다.

WSL 설치·업데이트가 필요하면 [WSL 설치 안내](https://learn.microsoft.com/ko-kr/windows/wsl/install)를 따릅니다. 이미 설치되어 있다면 PowerShell에서 다음 명령으로 업데이트합니다.

```powershell
wsl --update
```

NVIDIA GPU를 사용하려면 [드라이버](https://www.nvidia.co.kr/drivers)를 설치하고
[Docker Desktop의 GPU 지원 안내](https://docs.docker.com/desktop/features/gpu)를 확인합니다.

### Linux

리눅스는 각 배포판별 도커 설치 절차를 참조합니다.

[Docker Engine 설치 안내 (Install Docker Engine)](https://docs.docker.com/engine/install/)

### 도커 컨테이너 실행

아래 도커 명령은 **호스트 터미널**에서 실행합니다.

```sh
docker pull codebasic/pydeep
```

실습 파일을 보관할 폴더로 이동한 뒤 아래 명령을 실행합니다. 현재 폴더가 컨테이너의 `/workspace`에 연결됩니다. GPU 사용 여부와 포트 등은 환경에 맞게 조정합니다.

PowerShell:

```powershell
docker run --name pydeep --gpus=all --shm-size=2g -p 8888:8888 -v "${pwd}:/workspace" -d codebasic/pydeep
```

bash/zsh:

```bash
docker run --name pydeep --gpus=all --shm-size=2g -p 8888:8888 -v "$(pwd):/workspace" -d codebasic/pydeep
```

실행 후 웹브라우저에서 **[http://localhost:8888](http://localhost:8888)** 을 엽니다.

노트북에서 실습용 Python 커널을 선택합니다. 실습 파일은 `/workspace` 아래에 저장해야 호스트 폴더에 남습니다.

> [!NOTE]
> * **GPU:** NVIDIA GPU를 사용하지 않는 컴퓨터에서는 `--gpus=all`을 제외합니다.
> * **공유 메모리:** `--shm-size=2g`로 확보합니다. 필요에 따라 `--ipc=host`로 대체할 수 있습니다.
> * **외부 연결 차단:** 현재 컴퓨터에서만 접속하려면 `-p 127.0.0.1:8888:8888`을 사용합니다. 기본 설정은 외부에서 도달 가능한 환경에서 무인증 접속을 허용합니다. Compose에서도 포트에 `127.0.0.1`을 지정해 외부 연결을 차단합니다(예: `127.0.0.1:8888:8888`, `127.0.0.1:6006:6006`).
> * **인증 설정:** 인증이 필요한 경우 [Jupyter Server 공식 인증 설정 안내](https://jupyter-server.readthedocs.io/en/latest/operators/public-server.html#securing-a-jupyter-server)를 참조합니다.

#### 컨테이너 관리

이미 만든 컨테이너는 새로 만들지 않고 시작합니다.

```sh
docker start pydeep
```

기존 컨테이너에는 새 이미지를 받아도 자동 적용되지 않습니다. 교체 전 `/workspace` 바깥에 저장한 파일도 따로 보관하고,
기존 컨테이너를 중지·제거한 뒤 같은 작업 폴더로 위 실행 명령을 사용합니다.

#### 선택: 고급 설정·확인

저장소의 `compose.yaml`을 사용하는 경우 실습 폴더에 파일을 놓고 `docker compose up -d`로 시작할 수 있습니다.
현재 폴더를 `/workspace`에 연결하고 공유 메모리 4GB를 사용합니다. GPU가 없으면 `gpus: all`을 제외합니다.

* GPU 확인: 호스트에서 `docker exec pydeep nvidia-smi`를 실행합니다.
* 이미지 파일로 전달받은 경우: `docker load --input pydeep.tar` 후 태그를 확인합니다.
* 포트·공유 메모리·마운트 옵션: [포트 게시](https://docs.docker.com/engine/network/port-publishing/), [바인드 마운트](https://docs.docker.com/engine/storage/bind-mounts/), [공유 메모리](https://docs.docker.com/reference/cli/docker/container/run/#ipc)를 참조합니다.

## 코드 에이전트

1. 도커가 실행 중인 **호스트 터미널**에서 실행 중인 컨테이너에 진입합니다.

   ```sh
   docker exec -it pydeep bash
   ```

2. 진입한 **Linux 컨테이너 안에서** 원하는 코드 에이전트를 설치합니다. 아래 공식 문서의 **Linux 설치 절차**를 따릅니다. 호스트가 Mac이나 Windows여도 컨테이너에서는 Linux용 안내를 사용합니다.

   * [Claude Code 설치 안내](https://code.claude.com/docs/en/setup#install-claude-code)
   * [OpenAI Codex CLI 설치 안내](https://learn.chatgpt.com/docs/codex/cli)

3. 각 공식 문서에 따라 인증한 뒤 **같은 컨테이너 안에서** 코드 에이전트를 실행합니다.

설치 명령과 인증 방법은 변경될 수 있으므로 각 공식 문서의 최신 안내를 참조합니다.

## 로컬 모델

코드 에이전트는 기본적으로 클라우드 서비스를 이용합니다. 보안이나 비용 등의 이유로 클라우드 서비스를 사용하기 어렵다면 [Ollama](https://ollama.com)로 언어 모델을 로컬에서 실행하고, 채팅과 코드 에이전트에 연결할 수 있습니다.

앞 절처럼 **Linux 컨테이너 안에서** [Ollama를 설치](https://docs.ollama.com/linux)합니다.

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

설치 후, 서버를 백그라운드에서 실행합니다. 코드 에이전트에 연결하려면 모델 컨텍스트 크기가 64K 이상이어야 하므로, 서버를 시작할 때 `OLLAMA_CONTEXT_LENGTH` 환경 변수를 지정합니다.

```sh
OLLAMA_CONTEXT_LENGTH=65536 nohup ollama serve > /tmp/ollama.log 2>&1 < /dev/null &
```

컨테이너를 재시작하면 서버 실행 명령도 다시 실행합니다.

사용할 모델을 내려받습니다. 예:

```sh
ollama pull gemma4
```

### 채팅 인터페이스

[oterm](https://ggozad.github.io/oterm/features/)은 Markdown과 코드 서식을 보여주는 터미널 채팅 UI입니다. Ollama 기본 채팅 대신 사용할 수 있습니다. 서버 기동 후 **같은 컨테이너 안에서** 실행합니다.

```sh
uvx oterm
```

첫 실행 시 패키지를 내려받으며, `localhost:11434`의 Ollama를 자동 감지합니다. 화면에서 로컬 모델을 선택해 대화합니다.

### 코드 에이전트 연결

앞 절에서 설치한 코드 에이전트는 [`ollama launch`](https://docs.ollama.com/cli#launch-integrations)로 연결할 수 있습니다.

## 직접 설치

위의 도커 환경에 대응하는 Python 실습 환경입니다. Mac(Apple Silicon), Linux, Windows(WSL2)에서 사용할 수 있습니다. 실습 노트북을 보관할 **작업 폴더의 터미널**에서 실행하며, 해당 폴더의 `.venv`에 패키지와 JupyterLab을 함께 설치합니다.

### OS별 준비

#### Mac (Apple Silicon)

Homebrew를 준비한 뒤 실행합니다.

```sh
brew install uv libomp
```

#### Linux / Windows(WSL2) — Ubuntu 기준

Windows에서는 PowerShell이 아닌 WSL2의 Ubuntu 터미널에서 실행합니다.

```sh
sudo apt update
sudo apt install -y curl libgomp1
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"
```

### 공통 설치

```sh
uv venv --python 3.12
uv pip install \
  numpy scipy pandas \
  scikit-learn matplotlib \
  xgboost lightgbm \
  torch torchvision tensorflow keras \
  jupyterlab ipykernel ipywidgets
```

### JupyterLab 실행

```sh
uv run jupyter lab
```

### 유의사항

* Mac은 CUDA 대신 [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)를 사용하며 연산별 지원은 다를 수 있습니다. Mac의 XGBoost·LightGBM은 CPU로 사용합니다.
* Linux·WSL2의 NVIDIA GPU 가속에는 호환 드라이버와 프레임워크별 CUDA 조건이 필요합니다. [TensorFlow GPU 설치](https://www.tensorflow.org/install/pip)의 `tensorflow[and-cuda]` 추가는 PyTorch와 CUDA 의존성이 충돌할 수 있어 별도 호환성 확인이 필요하며, 위 공통 명령만으로 두 프레임워크의 GPU 동작을 보장하지 않습니다.
* TensorFlow의 Mac GPU 가속은 별도 [TensorFlow Metal](https://developer.apple.com/metal/tensorflow-plugin/) 플러그인과 버전 호환성 확인이 필요합니다.
* 직접 설치는 Python 환경만 도커 이미지에 대응하도록 구성했습니다. 운영체제별 상세 설정은 도커 이미지와 다를 수 있습니다.
* 작업 폴더에 이미 `.venv`가 있으면 설치 과정에서 갱신될 수 있으니 유의합니다.
