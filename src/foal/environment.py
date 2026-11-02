"""Environment and hardware metadata collection for experiment reproducibility."""
from __future__ import annotations
import platform
import subprocess
import sys
from typing import Any, Dict


def get_git_info() -> dict[str, Any]:
    info: dict[str, Any] = {'commit': None, 'branch': None, 'dirty': False}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         stderr=subprocess.DEVNULL, text=True).strip()
        info['commit'] = commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                         stderr=subprocess.DEVNULL, text=True).strip()
        info['branch'] = branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        status = subprocess.check_output(['git', 'status', '--porcelain'],
                                         stderr=subprocess.DEVNULL, text=True).strip()
        info['dirty'] = len(status) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return info


def get_environment_info() -> dict[str, Any]:
    """Capture library versions, OS, hardware, and Git metadata."""
    import torch
    import torchvision
    import timm

    try:
        import datasets
        datasets_version = datasets.__version__
    except ImportError:
        datasets_version = None

    try:
        import wandb
        wandb_version = wandb.__version__
    except ImportError:
        wandb_version = None

    cuda_available = torch.cuda.is_available()
    gpu_info = None
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            'count': torch.cuda.device_count(),
            'device_name': props.name,
            'total_memory_gb': round(props.total_memory / (1024**3), 2),
            'cuda_version': torch.version.cuda,
            'capability': (props.major, props.minor),
        }

    return {
        'os': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'python': {
            'version': sys.version,
            'implementation': platform.python_implementation(),
        },
        'libraries': {
            'torch': torch.__version__,
            'torchvision': torchvision.__version__,
            'timm': timm.__version__,
            'datasets': datasets_version,
            'wandb': wandb_version,
        },
        'hardware': {
            'cuda_available': cuda_available,
            'gpu': gpu_info,
            'cpu_count': platform.os.cpu_count() if hasattr(platform, 'os') else None,
        },
        'git': get_git_info(),
    }
