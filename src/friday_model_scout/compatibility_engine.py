"""Compatibility engine for model-scout."""

from typing import Dict, Any, List, Tuple
import re
from .hardware_scanner import DetailedHardwareProfile

# Tok/s Heuristic Constants
BASE_GPU_TOK_S = 30.0
APPLE_GPU_TOK_S = 15.0
CPU_TOK_S_FACTOR = 0.4
MOE_ACTIVE_PARAMS_RATIO = 0.2

def parse_params(params_str: str) -> float:
    """Parse parameter string (e.g., '8B', '8x7B') to a float value."""
    # Handle MoE like 8x7B
    if 'x' in params_str.lower():
        parts = params_str.lower().split('x')
        try:
            # For MoE, we often care about total params for VRAM but active for Tok/s
            # Here we return total params for VRAM estimation fallback
            return float(parts[0]) * float(parts[1].replace('b', ''))
        except (ValueError, IndexError):
            return 7.0 # Fallback
    
    # Handle standard B
    try:
        return float(params_str.lower().replace('b', ''))
    except ValueError:
        return 7.0 # Fallback

def calculate_compatibility(model: Dict[str, Any], profile: DetailedHardwareProfile) -> Dict[str, Any]:
    """Calculate compatibility metrics for a model and hardware profile."""
    vram_reqs = model["vram_req"]
    ram_req = model["ram_req"]
    
    # 1. Determine "Fit"
    fit = "Won't Run"
    best_quant = None
    vram_needed = 0.0
    
    # Check GPU first
    if profile.gpu_vram_gb:
        for quant in ["Q8_0", "Q5_K_M", "Q4_K_M"]:
            if profile.gpu_vram_gb >= vram_reqs[quant]:
                fit = "Perfect" if quant == "Q8_0" else "Good"
                best_quant = quant
                vram_needed = vram_reqs[quant]
                break
    
    # If not on GPU, check RAM (CPU fallback)
    if fit == "Won't Run" and profile.ram_gb >= ram_req:
        fit = "Good" # CPU only is "Good" if it fits in RAM
        best_quant = "Q4_K_M"
        vram_needed = vram_reqs["Q4_K_M"]
        
    # 2. Calculate "Score" (0-100)
    score = 0
    if fit != "Won't Run":
        if profile.gpu_vram_gb:
            # Ratio of available VRAM to Q4 requirement
            vram_ratio = profile.gpu_vram_gb / vram_reqs["Q4_K_M"]
            score = min(100, int(vram_ratio * 70)) # Cap at 100
            if fit == "Perfect":
                score = min(100, score + 30)
        else:
            # CPU only score
            ram_ratio = profile.ram_gb / ram_req
            score = min(60, int(ram_ratio * 40)) # Lower max score for CPU
            
    # 3. Est. Tok/s (Heuristic)
    tok_s = 0.0
    params_val = parse_params(model["params"])
    
    if fit != "Won't Run":
        if profile.gpu_vram_gb and profile.gpu_vram_gb >= vram_needed:
            # GPU Tok/s
            base = APPLE_GPU_TOK_S if "Apple" in (profile.gpu_name or "") else BASE_GPU_TOK_S
            
            # MoE adjustment for Tok/s (active params)
            active_params = params_val
            if 'x' in model["params"].lower():
                # Rough heuristic: active params is approx 20-25% of total for MoE
                active_params = params_val * MOE_ACTIVE_PARAMS_RATIO
            
            tok_s = max(1.0, base * (10 / max(1.0, active_params)))
        else:
            # CPU Tok/s
            tok_s = max(0.2, (profile.cpu_cores * CPU_TOK_S_FACTOR) * (4 / max(1.0, params_val)))

    return {
        "fit": fit,
        "score": score,
        "tok_s": round(tok_s, 1),
        "best_quant": best_quant or "N/A",
        "vram_needed": vram_needed
    }

def get_compatible_models(models: List[Dict[str, Any]], profile: DetailedHardwareProfile) -> List[Dict[str, Any]]:
    """Get all models with their compatibility metrics."""
    results = []
    for model in models:
        compat = calculate_compatibility(model, profile)
        results.append({**model, "compat": compat})
    return results
