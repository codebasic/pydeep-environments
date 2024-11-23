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
NUMPY_VERSION = os.getenv("NUMPY_VERSION", "1.26")
TENSORFLOW_VERSION = os.getenv("TENSORFLOW_VERSION", "2.17")
PYTORCH_VERSION = os.getenv("TORCH_VERSION", "2.5")
KERAS_VERSION = os.getenv("KERAS_VERSION", "3.6")
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
        "WINDOWS_TENSORFLOW": '\n'.join([
            'Windows에서는 Tensorflow CUDA 지원 GPU 가속이 어렵습니다.',
            '1. Windows의 WSL2 또는 Docker를 사용하여 GPU 가속을 활성화할 수 있습니다.',
            "2. PyTorch는 Windows에서 CUDA 지원 GPU 가속을 지원합니다.\n  PyTorch를 설치하려면 설치 스크립트에서 'pytorch'로 선택하세요.",
            '3. 설치를 계속 진행하면 CPU용 TensorFlow가 설치됩니다.',
        ]),
    },
    "USER_PROMPT": {'PROCEED': "진행하시겠습니까?", 'ABORT': "중단하시겠습니까?"},
    "ERROR": {
        "UNSUPPORTED_PLATFORM": '\n'.join([
            "지원되지 않는 플랫폼 또는 아키텍처입니다.",
            "지원되는 플랫폼:",
            "\t- windows: amd64",
            "\t- linux: x86_64",
            "\t- darwin: x86_64, arm64",
        ]),
        "CONDA_ENV_EXISTS": "이미 존재하는 Conda 환경입니다."
    },
    "INFO": {
        "PROCEED_INSTALL": "설치를 진행합니다.",
        "CREATE_CONDA_ENV": "Conda 환경 생성 중...",
        "ABORT": "작업이 중단되었습니다.",
        "REMOVE_ENV": "환경을 제거하려면 다음 명령어를 실행하세요: conda remove --name {env_name} --all",
        'BUILD_NUMPY': "NumPy {NUMPY_VERSION} 소스 빌드",
    }
}

commands = {
    'CONDA_INSTALL': ["conda", "install", "-y",],
    'CONDA_RUN': ["conda", "run", "--no-capture-output" ],
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

def normalize_version(version, style="pip"):
    """
    버전을 정규화하여 pip 또는 conda 스타일로 반환.
    Args:
        version (str): 사용자 입력 버전 (예: "2.17", "1.10.0", "~=2.17.1").
        style (str): "pip" 또는 "conda" 중 하나를 선택 (기본값: "pip").
    Returns:
        str: 정규화된 버전 (pip 스타일: "~=2.17.0", conda 스타일: "2.17").
    """
    parsed_version = Version(version)

    # 사용자가 정확한 패치 버전을 지정한 경우 그대로 반환
    if parsed_version.micro != 0:
        return version  # 사용자가 지정한 버전 존중

    # 메이저 및 마이너만 주어진 경우
    if style == "pip":
        return f"{parsed_version.major}.{parsed_version.minor}.0"  # pip 스타일: ~=x.y.0
    elif style == "conda":
        return f"{parsed_version.major}.{parsed_version.minor}"  # conda 스타일: x.y

    # 기본적으로 사용자 입력 그대로 반환
    return version

def install_tensorflow(env_name, gpu=None, **kwargs):
    """TensorFlow 및 관련 패키지 설치."""
    # 시스템 및 아키텍처 조건 확인
    package_name = "tensorflow"

    # 추가 키워드 인자 처리
    version = kwargs.get("version", TENSORFLOW_VERSION)
    version = normalize_version(version, "pip")
    CUDA_OVERRIDE = kwargs.get("cuda_override", False)

    if gpu == 'cuda' or CUDA_OVERRIDE:
        package_name += "[and-cuda]"
    elif gpu is None:
        package_name += "-cpu"
        
    run_command(commands['CONDA_RUN'] + ["-n", env_name, "--no-capture-output", "pip", "install", f"{package_name}~={version}"])

    if gpu == 'metal':
        print("Apple Silicon Metal 가속 사용을 위한 Tensorflow-Metal 설치")
        run_command(commands['CONDA_RUN'] + ["-n", env_name,  "pip", "install", "tensorflow-metal"])
  
def install_pytorch(env_name, gpu=None, **kwargs):
    """PyTorch 및 관련 패키지 설치."""
    # 추가 키워드 인자 처리
    version = kwargs.get("version", PYTORCH_VERSION)
    version = normalize_version(version, "conda")
    cuda_version = kwargs.get("cuda", CUDA_VERSION)
    cuda_version = normalize_version(cuda_version, "conda")
    CUDA_OVERRIDE = kwargs.get("cuda_override", False)

    channels = ["pytorch"]
    packages = [f"pytorch={version}", "torchvision", "torchaudio"]
    if gpu == 'cuda' or CUDA_OVERRIDE:
        channels.append("nvidia")
        packages.append(f"pytorch-cuda={cuda_version}")
    elif gpu is None:
        packages.append("cpuonly")
    
    run_command(commands['CONDA_INSTALL'] + ["-n", env_name,] + packages + format_conda_channels(channels))

    # Keras 설치
    run_command(commands['CONDA_RUN'] + ["-n", env_name,
        "pip", "install", f"keras~={normalize_version(KERAS_VERSION)}"
    ])

def build_numpy(env_name):
    """Build and install NumPy from source."""
    print(messages['INFO']['BUILD_NUMPY'].format(NUMPY_VERSION=normalize_version(NUMPY_VERSION)))
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

def add_jupyter_kernel(env_name):
    """특정 Conda 환경을 Jupyter 커널로 등록."""
    if check_conda_environment(env_name):
        print(f"환경 {env_name}을(를) Jupyter 커널로 등록합니다.")
        run_command(commands['CONDA_RUN'] + [
            "-n", env_name,
            "python", "-m", "ipykernel", "install", "--user",
            "--name", env_name,
            "--display-name", env_name.capitalize()
        ])
    else:
        print(f"환경 {env_name}이(가) 존재하지 않습니다. 먼저 환경을 생성하세요.")
        return 1
    return 0

def add_common_framework_arguments(parser, framework): 
    """TensorFlow와 PyTorch에 공통적인 명령줄 옵션을 추가합니다."""
    parser.add_argument("--build-numpy", action="store_true", help="NumPy를 소스로부터 빌드")
    if platform.uname().machine.lower() in ["x86_64", "amd64"]:
        parser.add_argument("--cuda-override", action="store_true", help="CUDA 감지 실패 시 강제로 CUDA 설치를 시도")
    parser.add_argument(
        "--version",
        type=str,
        default={'tensorflow': TENSORFLOW_VERSION, 'pytorch': PYTORCH_VERSION}[framework],
        help=f"설치할 {framework.capitalize()} 버전 (환경 변수: {framework.upper()}_VERSION)"
    )

def handle_framework_install(framework, args):
    """TensorFlow 및 PyTorch 환경 설정 처리."""
    print(f"{framework.capitalize()} {args.version} 환경 설정 시작...")

    # Windows에서 TensorFlow 경고 및 사용자 확인
    system = platform.uname().system.lower().strip()
    if framework == "tensorflow" and system == "windows":
        print(messages['WARNING']['WINDOWS_TENSORFLOW'])
        if confirm_proceed(messages['USER_PROMPT']['ABORT'], default="y"):
            print(messages['INFO']['ABORT'])
            sys.exit(1)

    # GPU 감지
    gpu = detect_gpu(*get_system_info(), args.cuda_override)

    # Conda 환경 확인
    if check_conda_environment(framework):
        print(messages['ERROR']['CONDA_ENV_EXISTS'])
        print(messages['INFO']['REMOVE_ENV'].format(env_name=framework))
        sys.exit(1)
        # NumPy 빌드 옵션이 있는 경우, NumPy만 설치 후 종료
        if args.build_numpy:
            build_numpy(framework)
            return
    else:
        # Conda 환경 생성
        print(messages['INFO']['CREATE_CONDA_ENV'])
        create_conda_environment(framework, PYDATA_PACKAGES)

    # 프레임워크 설치
    if framework == "tensorflow":
        gpu = None if system == "windows" else gpu
        cuda_override = False if system == "windows" else args.cuda_override
        install_tensorflow(framework, gpu, version=args.version, cuda_override=cuda_override)
    elif framework == "pytorch":
        install_pytorch(framework, gpu, version=args.version, cuda=args.cuda, cuda_override=args.cuda_override)

    # NumPy 소스 빌드
    if args.build_numpy:
        build_numpy(framework)

    print_usage_instructions(framework)

def dynamic_usage(subparsers):
    """
    동적으로 사용법을 생성합니다.
    Args:
        subparsers (argparse._SubParsersAction): 서브 명령어 정보
    Returns:
        str: 동적으로 생성된 사용법
    """
    usage_lines = []
    for command, parser in subparsers.choices.items():
        # 중복된 usage: 제거
        usage_line = f"  pydeep.py {command} {parser.format_usage().split('usage:')[1].strip()}"
        usage_lines.append(usage_line)
    return "\n".join(usage_lines)


class KoreanHelpFormatter(argparse.HelpFormatter):
    """argparse 기본 도움말 포맷을 한국어로 변경합니다."""
    def add_usage(self, usage, actions, groups, prefix=None):
        """'usage'를 '사용법'으로 변경."""
        if prefix is None:
            prefix = "사용법:\n"
        super().add_usage(usage, actions, groups, prefix)

    def start_section(self, heading):
        """기본 헤딩을 한국어로 변경."""
        translations = {
            "positional arguments": "명령어",
            "optional arguments": "옵션",
        }
        heading = translations.get(heading, heading)
        super().start_section(heading)


def main():
    parser = argparse.ArgumentParser(
        description="딥러닝 환경 설정 스크립트", 
        formatter_class=KoreanHelpFormatter, 
        add_help=False)  # 기본 도움말 비활성화
    subparsers = parser.add_subparsers(dest="command")

    # 사용자 정의 도움말 추가
    parser.add_argument(
        "-h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="도움말을 표시하고 종료합니다."
    )

    # TensorFlow 명령어 추가
    tensorflow_parser = subparsers.add_parser("tensorflow", help="TensorFlow 환경 설정")
    add_common_framework_arguments(tensorflow_parser, 'tensorflow')

    # PyTorch 명령어 추가
    pytorch_parser = subparsers.add_parser("pytorch", help="PyTorch 환경 설정")
    add_common_framework_arguments(pytorch_parser, 'pytorch')
    # PyTorch 전용 옵션 추가
    if platform.uname().machine.lower() in ["x86_64", "amd64"]:
        pytorch_parser.add_argument(
            "--cuda",
            type=str,
            default=CUDA_VERSION,  # 글로벌 변수에서 기본값 설정
            help=f"PyTorch CUDA 버전 (기본값: {CUDA_VERSION}). PyTorch에만 적용됩니다. (x86 아키텍처에서만 사용 가능)"
        )

    # Jupyter 명령어 추가
    jupyter_parser = subparsers.add_parser("jupyter", help="Jupyter 환경 관리")
    jupyter_parser.add_argument(
        "action", 
        choices=["install", "add"], 
        help="Jupyter 환경 설치(install) 또는 커널 등록(add)"
    )

    # 'add' 명령에만 필요한 옵션 추가
    jupyter_parser.add_argument(
        "--env",
        choices=["tensorflow", "pytorch"],
        required=False,  # 'add'에서만 필요하므로 필수로 지정하지 않음
        help="Jupyter 커널로 등록할 환경 이름"
    )

    # 동적 사용법 생성
    parser.usage = dynamic_usage(subparsers)

    args = parser.parse_args()

    if args.command == "tensorflow":
        handle_framework_install("tensorflow", args)
    elif args.command == "pytorch":
        handle_framework_install("pytorch", args)
    elif args.command == "jupyter":
        if args.action == "install":
            print("Jupyter 환경 설치 중...")
            create_conda_environment("jupyter", ["jupyterlab", "jupyterlab_widgets", "-c", "conda-forge"])
            print_usage_instructions("jupyter")
        elif args.action == "add":
            if not args.env:
                print("Error: '--env' 옵션으로 커널로 등록할 환경을 지정하세요 (예: --env tensorflow 또는 --env pytorch).")
                sys.exit(1)
            add_jupyter_kernel(args.env)
    else:
        parser.print_help()

if __name__ == "__main__":
    print_banner()
    main()
