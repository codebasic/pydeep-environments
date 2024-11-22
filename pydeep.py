import argparse
import subprocess
import os, sys, platform
from datetime import datetime
from packaging.version import Version

def print_banner():
    """스크립트 실행 시 배너 헤드를 출력합니다."""
    banner = r"""
 ██████╗ ██████╗ ██████╗ ███████╗██████╗  █████╗ ███████╗██╗ ██████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██║██╔════╝
██║     ██║   ██║██║  ██║█████╗  ██████╔╝███████║███████╗██║██║     
██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗██╔══██║╚════██║██║██║     
╚██████╗╚██████╔╝██████╔╝███████╗██████╔╝██║  ██║███████║██║╚██████╗
 ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝
"""
    print(f"\033[1;36m{banner}\033[0m")
    print(f"\033[1;32m코드베이직 (c) 2015-{datetime.now().year}\033[0m")
    print("딥러닝 환경 설정 스크립트\n")

# Global variables
PYTHON_VERSION = os.getenv("PYTHON_VERSION", "3.10")
NUMPY_VERSION = os.getenv("NUMPY_VERSION", "1.26.0")
TENSORFLOW_VERSION = os.getenv("TENSORFLOW_VERSION", "2.17.0")
TORCH_VERSION = os.getenv("TORCH_VERSION", "2.5.0")
KERAS_VERSION = os.getenv("KERAS_VERSION", "3.6.0")
CUDA_VERSION = os.getenv("CUDA_VERSION", "11.8")

PYDATA_PACKAGES = [
    "pandas",
    "scikit-learn",
    "matplotlib",
    "pydot",
    "ipykernel",
    "ipywidgets",
    "sentencepiece"
]

# 지원되는 플랫폼 및 아키텍처
SUPPORTED_SYSTEMS = {
    "windows": ["amd64"],
    "linux": ["x86_64"],
    "darwin": ["x86_64", "arm64"]
}

# 메시지 딕셔너리
messages = {
    "WARNING": {
        "NO_GPU": "GPU가 감지되지 않았습니다.",
        "WINDOWS_TENSORFLOW": """
Windows에서는 Tensorflow CUDA 지원 GPU 가속이 어렵습니다.
1. Windows의 WSL2 또는 Docker를 사용하여 GPU 가속을 활성화할 수 있습니다.
2. PyTorch는 Windows에서 CUDA 지원 GPU 가속을 지원합니다. 
   PyTorch를 설치하려면 설치 스크립트에서 'pytorch'로 선택하세요.
3. 설치를 계속 진행하면 CPU용 TensorFlow가 설치됩니다.
"""
    },
    "USER_PROMPT": {'PROCEED': "진행하시겠습니까?", 'ABORT': "중단하시겠습니까?"},
    "ERROR": {
        "UNSUPPORTED_PLATFORM": """
지원되지 않는 플랫폼 또는 아키텍처입니다.
지원되는 플랫폼:
  - windows: amd64
  - linux: x86_64
  - darwin: x86_64, arm64
""",
        "CONDA_ENV_EXISTS": "이미 존재하는 Conda 환경입니다."
    },
    "INFO": {
        "PROCEED_INSTALL": "설치를 진행합니다.",
        "CONDA_ENV_CREATED": "Conda 환경 생성 중...",
        "ABORT": "작업이 중단되었습니다.",
        "REMOVE_ENV": "환경을 제거하려면 다음 명령어를 실행하세요: conda remove --name {env_name} --all",
        "USAGE": """
%(prog)s [tensorflow | pytorch] [--build-numpy] [--cuda-override]

사용 예:
  %(prog)s tensorflow                TensorFlow 환경 생성
  %(prog)s pytorch                   PyTorch 환경 생성
  %(prog)s tensorflow --build-numpy  TensorFlow 환경 생성 및 NumPy 소스 빌드
  %(prog)s pytorch --cuda-override   GPU 감지 실패 시 강제로 CUDA 설치 시도
""",
    }
}

def is_supported_platform(system, machine):
    """
    시스템 및 아키텍처가 지원되는지 확인합니다.
    """
    if system in SUPPORTED_SYSTEMS and machine in SUPPORTED_SYSTEMS[system]:
        return True
    return False

def get_system_info():
    """플랫폼 및 아키텍처 정보를 반환."""
    system_info = platform.uname()
    return system_info.system.lower(), system_info.machine.lower()

def is_cuda_available():
    try:
        result = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
        return "CUDA Version" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    
def detect_gpu(system, machine, cuda_override=False):
    """GPU 감지 및 설치 경로 결정."""
    gpu = 'cuda' if cuda_override else None
    if system == "linux" or system == "windows":
        # Linux 및 Windows에서 CUDA 감지
        gpu = 'cuda' if is_cuda_available() else None
    elif system == "darwin" and machine == "arm64":
        # macOS Apple Silicon에서 Metal 감지
        gpu = 'metal'

    # GPU 감지 실패 처리
    if gpu is None:
        print(messages['WARNING']['NO_GPU'])
    return gpu

def run_command(command, capture_output=False):
    """Run a shell command and handle errors."""
    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, text=True, capture_output=capture_output)
        if capture_output:
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed - {' '.join(command)}")
        sys.exit(e.returncode)

format_conda_channels = lambda channels: [item for channel in channels for item in ["-c", channel]]

def create_conda_environment(env_name, packages=[]):
    """Create a new Conda environment."""
    command = [
        "conda", "create", "-y", "-n", env_name,
        f"python={PYTHON_VERSION}"
    ] + packages
    run_command(command)

def check_conda_environment(env_name):
    """Check if a Conda environment already exists."""
    output = run_command(["conda", "info", "--envs"], capture_output=True)
    return any(env_name in line.split() for line in output.splitlines())

def install_tensorflow(env_name, gpu=None):
    """TensorFlow 및 관련 패키지 설치."""
    # 시스템 및 아키텍처 조건 확인
    package_name = "tensorflow"
    if gpu == 'cuda':
        package_name += "[and-cuda]"
    elif gpu is None:
        package_name += "-cpu"
        
    run_command(["conda", "run", "-n", env_name, "pip", "install", f"{package_name}~={TENSORFLOW_VERSION}"])

    if gpu == 'metal':
        print("Apple Silicon Metal 가속 사용을 위한 Tensorflow-Metal 설치")
        run_command(["conda", "run", "-n", env_name, "pip", "install", "tensorflow-metal"])
  
def install_pytorch(env_name, gpu=None):
    """PyTorch 및 관련 패키지 설치."""
    channels = ["pytorch"]
    packages = ["pytorch", "torchvision", "torchaudio"]
    if gpu == 'cuda':
        channels.append("nvidia")
        packages.append(f"pytorch-cuda={CUDA_VERSION}")
    
    run_command(["conda", "install", "-y", "-n", env_name,] + packages + format_conda_channels(channels))

    # Keras 설치
    run_command([
        "conda", "run", "--no-capture-output", "-n", env_name,
        "pip", "install", f"keras~={KERAS_VERSION}"
    ])

def build_numpy(env_name):
    """Build and install NumPy from source."""
    print(f"NumPy {NUMPY_VERSION} 소스 빌드 ...")
    run_command(["conda", "run", "--no-capture-output" , "-n", env_name, "pip", "install", "--force-reinstall", "--no-binary", ":all:", f"numpy~={NUMPY_VERSION}"])

def print_usage_instructions(env_name):
    """Conda 환경 사용 안내를 출력합니다."""
    print("\n설치가 성공적으로 완료되었습니다!")
    print("==================================================")
    print(f"Conda 환경 '{env_name}'을(를) 사용하려면 다음 단계를 따르세요:")
    print(f"  1. 환경 활성화: conda activate {env_name}")
    print(f"  2. 환경 내에서 필요한 작업 수행")
    print(f"  3. 환경 비활성화: conda deactivate")
    print("\n환경을 제거하려면 다음 명령어를 실행하세요:")
    print(f"  conda remove --name {env_name} --all")
    print("==================================================")

def confirm_proceed(user_prompt=messages['USER_PROMPT']['PROCEED'], default="y"):
    """
    사용자에게 작업을 진행할지 여부를 묻고, y/n 입력을 처리합니다.
    Args:
        user_prompt (str): 사용자에게 표시할 메시지.
        default (str): 기본값 ("y" 또는 "n").
    Returns:
        bool: 사용자가 "y"를 입력했는지 여부.
    """
    valid_responses = {"y": True, "n": False, "yes": True, "no": False}
    default = default.lower()
    assert default in valid_responses, "default 값은 'y' 또는 'n'이어야 합니다."

    # 메시지 생성
    message = f"{user_prompt} (y/[n]): " if default == "n" else f"{user_prompt} ([y]/n): "

    # 사용자 입력 처리
    user_input = input(message).strip().lower()
    if not user_input:
        user_input = default
    return valid_responses.get(user_input, False)

def main():
    parser = argparse.ArgumentParser(description="코드베이직 딥러닝 환경 설정", usage=messages['INFO']['USAGE'])
    parser.add_argument("env_name", choices=["tensorflow", "pytorch"], help="생성할 환경 명칭")
    parser.add_argument("--build-numpy", action="store_true", help="NumPy를 소스로부터 빌드")
    parser.add_argument(
        "--cuda-override", 
        action="store_true", 
        help="CUDA 감지 실패 시 강제로 CUDA 설치를 시도"
    )
    args = parser.parse_args()

    env_name = args.env_name
    build_numpy_flag = args.build_numpy

    # 시스템 및 아키텍처 정보 확인
    system, machine = get_system_info()
    print(f"{system} {machine}")

    # 지원 여부 확인
    if not is_supported_platform(system, machine):
        print(messages['ERROR']['UNSUPPORTED_PLATFORM'])
        sys.exit(1)
    
    # Windows에서 TensorFlow 설치 경고 및 진행 여부 확인
    elif system == "windows" and env_name == "tensorflow":
        print(messages['WARNING']['WINDOWS_TENSORFLOW'])
        # 중단 여부 확인
        if confirm_proceed(messages['USER_PROMPT']['ABORT'], default="y"):
            print(messages['INFO']['ABORT'])
            sys.exit(0)

    # GPU 감지
    gpu = detect_gpu(system, machine, args.cuda_override)
    if gpu is None and not confirm_proceed(messages['USER_PROMPT']['ABORT'], default="y"):
        print(messages['INFO']['ABORT'])
        sys.exit(0)

    # 기존 환경 확인. 이미 존재하는 경우 설치 중단
    if check_conda_environment(env_name):
        print(f"{messages['ERROR']['CONDA_ENV_EXISTS']} '{env_name}'")
        print(messages['INFO']['ABORT'])
        print(messages['INFO']['REMOVE_ENV'].format(env_name=env_name))
        sys.exit(1)
    
    # 새로운 환경 생성
    print(messages['INFO']['PROCEED_INSTALL'])
    create_conda_environment(env_name, PYDATA_PACKAGES)

    # Install frameworks
    if env_name == "tensorflow":
        print(f"TensorFlow {TENSORFLOW_VERSION} {messages['INFO']['PROCEED_INSTALL']}")
        if system == "windows":
            gpu = None
        install_tensorflow(env_name, gpu)

    elif env_name == "pytorch":
        print(f"PyTorch {TORCH_VERSION} {messages['INFO']['PROCEED_INSTALL']}")
        install_pytorch(env_name, gpu) 

    # Build NumPy if requested
    if build_numpy_flag:
        build_numpy(env_name)

    # Print usage instructions
    print_usage_instructions(env_name)

if __name__ == "__main__":
    print_banner()
    main()
