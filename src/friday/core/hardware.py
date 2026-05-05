"""Hardware profiler for auto-detecting system resources."""

import asyncio
import logging
import platform
from dataclasses import dataclass
from typing import Any

import psutil

try:
    import GPUtil
except ImportError:
    GPUtil = None

logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    os: str
    cpu_arch: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpu_vram_gb: float | None
    gpu_name: str | None

    def to_detailed(self) -> Any:
        """Map Friday's HardwareProfile to Model Scout's DetailedHardwareProfile."""
        from friday_model_scout.hardware_scanner import DetailedHardwareProfile
        return DetailedHardwareProfile(
            os=self.os,
            cpu_arch=self.cpu_arch,
            cpu_cores=self.cpu_cores,
            cpu_threads=self.cpu_threads,
            ram_gb=self.ram_gb,
            gpu_name=self.gpu_name,
            gpu_vram_gb=self.gpu_vram_gb
        )

# Hardware-to-model mapping heuristics
# Ordered from most capable to least capable
MODEL_RECOMMENDATIONS = [
    {"vram_min": 12.0, "model": "llama3:70b"},
    {"vram_min": 6.0, "model": "llama3:8b"},
    {"ram_min": 8.0, "model": "phi3:mini"},
]
DEFAULT_MODEL = "gemma2:2b"

def get_recommended_model(profile: HardwareProfile) -> str:
    """Return a suggested Ollama model based on hardware capabilities."""
    # Check VRAM-based recommendations first (GPU)
    if profile.gpu_vram_gb:
        for rec in MODEL_RECOMMENDATIONS:
            if "vram_min" in rec and profile.gpu_vram_gb > rec["vram_min"]:
                return rec["model"]
    
    # Fallback to RAM-based recommendations (CPU)
    for rec in MODEL_RECOMMENDATIONS:
        if "ram_min" in rec and profile.ram_gb > rec["ram_min"]:
            return rec["model"]
            
    return DEFAULT_MODEL

async def get_hardware_profile() -> HardwareProfile:
    """Detect and return hardware profile."""
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_cores = psutil.cpu_count(logical=False) or 0
    cpu_threads = psutil.cpu_count(logical=True) or 0
    cpu_arch = platform.machine()
    
    gpu_vram_gb = None
    gpu_name = None
    
    if GPUtil:
        try:
            # SECURE: Safe check if nvidia-smi responds quickly
            if platform.system() != "Windows":
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "nvidia-smi", "-L",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("GPU detection timed out. nvidia-smi is hanging.")
            
            gpus = await asyncio.to_thread(GPUtil.getGPUs)
            if gpus:
                # Take the best GPU
                gpu = gpus[0]
                # GPUtil memoryTotal is in MB
                gpu_vram_gb = gpu.memoryTotal / 1024
                gpu_name = gpu.name
        except Exception as e:
            logger.warning(f"Failed to detect GPU: {e}")

    # Fallback/Additional detection for macOS (Apple Silicon)
    if platform.system() == "Darwin" and not gpu_name:
        try:
            # Check for Apple Silicon
            proc = await asyncio.create_subprocess_exec(
                "sysctl", "-n", "machdep.cpu.brand_string",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            cpu_info = stdout.decode().strip()
            
            if "Apple" in cpu_info:
                gpu_name = cpu_info
                # On Apple Silicon, RAM is unified, so we can consider a portion as VRAM
                # Typically macOS allows up to ~75% of RAM for GPU
                gpu_vram_gb = ram_gb * 0.75 
        except Exception:
            pass

    return HardwareProfile(
        os=platform.system(),
        cpu_arch=cpu_arch,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        gpu_vram_gb=gpu_vram_gb,
        gpu_name=gpu_name
    )
