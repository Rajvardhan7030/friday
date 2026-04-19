"""Hardware profiler for auto-detecting system resources."""

import psutil
import platform
import logging
import subprocess
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    import GPUtil
except ImportError:
    GPUtil = None

logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    os: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpu_vram_gb: Optional[float]
    gpu_name: Optional[str]

    def recommend_model(self) -> str:
        """Recommend a model based on hardware profile."""
        if self.gpu_vram_gb and self.gpu_vram_gb >= 8:
            return "qwen2.5-coder:7b"
        elif self.ram_gb >= 16:
            return "llama3.1:8b"
        elif self.ram_gb >= 8:
            return "phi4"
        else:
            return "tinyllama"

def get_hardware_profile() -> HardwareProfile:
    """Detect and return hardware profile."""
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_cores = psutil.cpu_count(logical=False) or 0
    cpu_threads = psutil.cpu_count(logical=True) or 0
    
    gpu_vram_gb = None
    gpu_name = None
    
    if GPUtil:
        try:
            # SECURE: Safe check if nvidia-smi responds quickly
            if platform.system() != "Windows":
                subprocess.run(
                    ["nvidia-smi", "-L"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    timeout=2.0
                )
            
            gpus = GPUtil.getGPUs()
            if gpus:
                # Take the best GPU
                gpu = gpus[0]
                # GPUtil memoryTotal is in MB
                gpu_vram_gb = gpu.memoryTotal / 1024
                gpu_name = gpu.name
        except subprocess.TimeoutExpired:
            logger.warning("GPU detection timed out. nvidia-smi is hanging.")
        except Exception as e:
            logger.warning(f"Failed to detect GPU: {e}")

    return HardwareProfile(
        os=platform.system(),
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        gpu_vram_gb=gpu_vram_gb,
        gpu_name=gpu_name
    )
