"""Hardware scanner for model-scout."""

import psutil
import platform
import logging
import subprocess
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import GPUtil
except ImportError:
    GPUtil = None

logger = logging.getLogger(__name__)

@dataclass
class DetailedHardwareProfile:
    os: str
    cpu_arch: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None

def scan_hardware() -> DetailedHardwareProfile:
    """Detect and return a detailed hardware profile."""
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_cores = psutil.cpu_count(logical=False) or 0
    cpu_threads = psutil.cpu_count(logical=True) or 0
    cpu_arch = platform.machine()
    
    gpu_vram_gb = None
    gpu_name = None
    
    # Try to detect GPU using GPUtil (NVIDIA)
    if GPUtil:
        try:
            # Quick check for nvidia-smi to avoid hangs
            if platform.system() != "Windows":
                subprocess.run(
                    ["nvidia-smi", "-L"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    timeout=2.0
                )
            
            gpus = GPUtil.getGPUs()
            if gpus:
                # Take the best GPU (highest VRAM)
                gpu = max(gpus, key=lambda x: x.memoryTotal)
                gpu_vram_gb = gpu.memoryTotal / 1024
                gpu_name = gpu.name
        except (subprocess.TimeoutExpired, Exception):
            pass

    # Fallback/Additional detection for macOS (Apple Silicon)
    if platform.system() == "Darwin" and not gpu_name:
        try:
            # Check for Apple Silicon
            cpu_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            if "Apple" in cpu_info:
                gpu_name = cpu_info
                # On Apple Silicon, RAM is unified, so we can consider a portion as VRAM
                # Typically macOS allows up to ~75% of RAM for GPU
                gpu_vram_gb = ram_gb * 0.75 
        except Exception:
            pass

    return DetailedHardwareProfile(
        os=platform.system(),
        cpu_arch=cpu_arch,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb
    )
