"""Check the local VisionPort Python environment."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_PACKAGES = ("streamlit", "torch", "ultralytics", "PIL")


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    failed = False
    for package in REQUIRED_PACKAGES:
        distribution = "pillow" if package == "PIL" else package
        try:
            installed_version = version(distribution)
            __import__(package)
        except (ImportError, PackageNotFoundError) as error:
            print(f"[FAIL] {package}: {error}")
            failed = True
        else:
            print(f"[ OK ] {package} {installed_version}")

    model_path = PROJECT_ROOT / "yolov8n.pt"
    if model_path.is_file():
        print(f"[ OK ] Model file: {model_path.name}")
    else:
        print(f"[FAIL] Model file missing: {model_path}")
        failed = True

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
