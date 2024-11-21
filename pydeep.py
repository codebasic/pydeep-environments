import argparse
import subprocess
import os, sys, platform
from packaging.version import Version

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

def create_conda_environment(env_name, packages=[]):
    """Create a new Conda environment."""
    print(f"Conda 환경 '{env_name}' 생성 ...")
    command = [
        "conda", "create", "-y", "-n", env_name,
        f"python={PYTHON_VERSION}"
    ] + packages
    run_command(command)

def check_conda_environment(env_name):
    """Check if a Conda environment already exists."""
    output = run_command(["conda", "info", "--envs"], capture_output=True)
    return any(env_name in line.split() for line in output.splitlines())

def get_system_info():
    """플랫폼 및 아키텍처 정보를 반환."""
    system_info = platform.uname()
    return system_info.system.lower(), system_info.machine.lower()

def install_tensorflow(env_name):
    """TensorFlow 및 관련 패키지 설치."""
    print(f"TensorFlow {TENSORFLOW_VERSION} 환경 설치를 시작합니다.")
    print("가능한 경우, GPU 가속을 위해 필요한 소프트웨어들이 같이 설치됩니다.")

    # 시스템 및 아키텍처 정보 확인
    system, machine = get_system_info()
    print(f"감지된 플랫폼: {system}, 아키텍처: {machine}")

    # 시스템 및 아키텍처 조건 확인
    if system == "darwin" and "arm" in machine:
        print("Apple Silicon Metal 가속을 사용하여 TensorFlow를 설치합니다.")
        run_command(["conda", "run", "-n", env_name, "pip", "install", f"tensorflow~={TENSORFLOW_VERSION}"])
        run_command(["conda", "run", "-n", env_name, "pip", "install", "tensorflow-metal"])

    elif system == "darwin" and "x86_64" in machine:
        print("Intel Mac: CPU를 위한 TensorFlow를 설치합니다.")
        run_command(["conda", "run", "-n", env_name, "pip", "install", f"tensorflow-cpu~={TENSORFLOW_VERSION}"])

    elif system == "linux" and "x86_64" in machine:
        print("CUDA 지원 GPU용 TensorFlow를 설치합니다.")
        package_name = "tensorflow[and-cuda]"
        run_command(["conda", "run", "-n", env_name, "pip", "install", f"{package_name}~={TENSORFLOW_VERSION}"])

    elif system == "windows" and "x86_64" in machine:
        if Version(TENSORFLOW_VERSION) > Version("2.10"):
            print(f"TensorFlow {TENSORFLOW_VERSION}에서는 윈도우즈 직접 설치 GPU 지원이 중단되었습니다.")
            print('Windows의 WSL2 또는 Docker를 사용하여 GPU 가속을 사용할 수 있습니다.')
            # 설치 중단 여부 확인            
            설치진행 = input("그래도 계속하시겠습니까? (y/[n]): ")
            if 설치진행.strip().lower() != "y":
                print("설치를 중단합니다.")
                sys.exit(0)
            
            run_command(["conda", "run", "-n", env_name, "pip", "install", f"tensorflow-cpu~={TENSORFLOW_VERSION}"])
        else:
            print(f"TensorFlow {TENSORFLOW_VERSION}용 CUDA Toolkit 및 cuDNN을 설치 중입니다.")
            run_command(["conda", "install", "-y", "-n", env_name, "-c", "conda-forge", "cudatoolkit=11.2", "cudnn=8.1"])
            run_command(["conda", "run", "-n", env_name, "pip", "install", f"tensorflow~={TENSORFLOW_VERSION}"])

    else:
        print(f"Error: 지원되지 않는 플랫폼 또는 아키텍처: {system}, {machine}")
        sys.exit(1)

def install_pytorch(env_name):
    """PyTorch 및 관련 패키지 설치."""
    print(f"PyTorch {TORCH_VERSION} 환경 설치를 시작합니다.")
    
    # 시스템 및 아키텍처 정보 확인
    system, machine = get_system_info()
    print(f"감지된 플랫폼: {system}, 아키텍처: {machine}")

    if system == "darwin":
        # macOS 설치
        run_command([
            "conda", "install", "-y", "-n", env_name, "-c", "pytorch",
            "pytorch", "torchvision", "torchaudio"
        ])
    elif machine == "x86_64":
        # x86_64 공통 설치 (Linux, Windows)
        print(f"PyTorch {TORCH_VERSION}용 CUDA {CUDA_VERSION} 및 cuDNN 설치")
        run_command([
            "conda", "install", "-y", "-n", env_name,
            "pytorch", "torchvision", "torchaudio", f"pytorch-cuda={CUDA_VERSION}",
            "-c", "pytorch", "-c", "nvidia"
        ])
    else:
        # 지원되지 않는 아키텍처 처리
        print(f"Error: 지원되지 않는 플랫폼 또는 아키텍처: {system}, {machine}")
        sys.exit(1)

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

def main():
    parser = argparse.ArgumentParser(description="코드베이직 딥러닝 환경 설정")
    parser.add_argument("env_name", choices=["tensorflow", "pytorch"], help="생성할 환경 명칭")
    parser.add_argument("--build-numpy", action="store_true", help="NumPy를 소스로부터 빌드")
    args = parser.parse_args()

    env_name = args.env_name
    build_numpy_flag = args.build_numpy

    if check_conda_environment(env_name):
        print(f"Error: 이미 존재하는 Conda 환경 '{env_name}'\n설치를 중단합니다.")
        sys.exit(1)

    # Create Conda environment
    create_conda_environment(env_name, PYDATA_PACKAGES)

    # Install frameworks
    if env_name == "tensorflow":
        install_tensorflow(env_name)
    elif env_name == "pytorch":
        install_pytorch(env_name) 

    # Build NumPy if requested
    if build_numpy_flag:
        build_numpy(env_name)

    # Print usage instructions
    print_usage_instructions(env_name)

if __name__ == "__main__":
    main()
