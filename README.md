# 딥러닝 환경 설정

Codebasic (c) 2015-2025

Windows와 Linux에서는 Docker를, Mac에서는 직접 설치를 권장합니다.

## Docker

아래 Docker 명령은 **호스트 터미널**에서 실행합니다.

### Windows

[Docker Desktop for Windows 설치 안내](https://docs.docker.com/desktop/install/windows-install)에 따라 WSL 2 기반으로 설치하고 Docker Desktop을 실행합니다.
요구사항과 라이선스 조건은 공식 문서를 확인하세요.

WSL 설치·업데이트가 필요하면 [WSL 설치 안내](https://learn.microsoft.com/ko-kr/windows/wsl/install)를 따릅니다.

NVIDIA GPU를 사용하려면 [드라이버](https://www.nvidia.co.kr/drivers)를 설치하고
[Docker Desktop의 GPU 지원 안내](https://docs.docker.com/desktop/features/gpu)를 확인합니다.

### Linux

리눅스는 각 배포판별 도커 설치 절차를 참조하시기 바랍니다.

[Docker Engine 설치 안내 (Install Docker Engine)](https://docs.docker.com/engine/install/)

### 도커 컨테이너 실행

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
공유 메모리는 `--shm-size=2g`로 확보합니다. Linux에서는 필요에 따라 `--ipc=host`로 대체할 수 있습니다.

실행 후 웹브라우저에서 **[http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab)** 을 엽니다.
JupyterLab은 자동으로 시작되며 토큰과 비밀번호 없이 접속합니다.
Windows에서 `localhost` 접속이 지연되면 위의 `127.0.0.1` 주소를 사용하세요.

> 위 예시는 현재 컴퓨터에서만 접속할 수 있습니다. `-p 8888:8888`로 변경하면 무인증 서버가 외부에 노출될 수 있습니다.

노트북에서 실습용 Python 커널을 선택합니다. 실습 파일은 `/workspace` 아래에 저장해야 호스트 폴더에 남습니다.

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
현재 폴더를 `/workspace`에 연결하고 공유 메모리 4GB를 사용합니다. GPU가 없으면 `gpus: all`을 제외합니다.

* GPU 확인: 호스트에서 `docker exec pydeep nvidia-smi`를 실행합니다.
* 이미지 파일로 전달받은 경우: `docker load --input pydeep.tar` 후 태그를 확인합니다.
* 포트·공유 메모리·마운트 옵션: [포트 게시](https://docs.docker.com/engine/network/port-publishing/), [바인드 마운트](https://docs.docker.com/engine/storage/bind-mounts/), [공유 메모리](https://docs.docker.com/reference/cli/docker/container/run/#ipc)를 참조합니다.

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

아래 명령은 Docker를 사용하지 않고 **호스트 터미널**에서 직접 환경을 구성하는 경우에 실행합니다.

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

###### MPS (Mac)

Mac에서는 Metal Performance Shaders(MPS)를 사용합니다.

```sh
conda activate pytorch
python -c "import torch; print('PyTorch Version:', torch.__version__); assert torch.backends.mps.is_available(), 'MPS not available'; print('MPS Device:', 'Apple Metal Performance Shaders')"
```

#### TensorFlow

* 설치 안내: [Tensorflow](https://www.tensorflow.org/install) 공식 문서를 참조해 OS/하드웨어에 맞게 설치합니다.
* 환경 생성: 공통 환경(`pyml`)을 복제해 TensorFlow 전용 환경을 만듭니다.

```sh
conda create --name tensorflow --clone pyml
```

### JupyterLab (선택)

[Astral uv 설치 안내](https://docs.astral.sh/uv/)에 따라 uv를 설치한 뒤 JupyterLab을 실행합니다.

```sh
uvx --from jupyterlab jupyter-lab
```

#### 주피터 커널

주피터에서 파이썬 환경을 사용하려면 각 환경을 주피터 커널(kernel)로 등록해야 합니다. 각 환경을 활성화 후, 다음 명령을 실행합니다.

예: PyTorch 환경 등록

```sh
conda activate pytorch
python -m ipykernel install --user --name pytorch --display-name "Pytorch 2"
```
