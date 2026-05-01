"""CLI for model-scout."""

import json
import sys
import asyncio
from typing import Optional
from .hardware_scanner import scan_hardware
from .model_database import get_models
from .compatibility_engine import get_compatible_models
from .tui import run_tui

async def run_scout(
    json_output: bool = False,
    filter_tag: Optional[str] = None,
    sort_by: str = "score",
    ollama_only: bool = False
):
    """Main entry point for the model-scout CLI."""
    
    # 1. Scan Hardware
    profile = scan_hardware()
    
    # 2. Get Models & Calculate Compatibility
    models = get_models()
    if ollama_only:
        models = [m for m in models if m.get("ollama_name")]
        
    results = get_compatible_models(models, profile)
    
    # 3. Output
    if json_output:
        output = {
            "hardware": {
                "os": profile.os,
                "cpu_arch": profile.cpu_arch,
                "cpu_cores": profile.cpu_cores,
                "ram_gb": profile.ram_gb,
                "gpu": {
                    "name": profile.gpu_name,
                    "vram_gb": profile.gpu_vram_gb
                }
            },
            "models": results
        }
        print(json.dumps(output, indent=2))
    else:
        # Filtering for initial TUI state if provided
        if filter_tag:
            results = [r for r in results if filter_tag.lower() in [t.lower() for t in r["tags"]]]
        
        # Launch Textual TUI
        await run_tui(profile, results)

def main():
    """Simple wrapper for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Friday Model Scout - LLM Hardware Compatibility Tool")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--filter", type=str, help="Filter models by tag")
    parser.add_argument("--sort", type=str, default="score", choices=["score", "name", "tok_s"], help="Sort by column (for JSON output)")
    parser.add_argument("--ollama-only", action="store_true", help="Only show models available on Ollama")
    
    args = parser.parse_args()
    
    asyncio.run(run_scout(
        json_output=args.json,
        filter_tag=args.filter,
        sort_by=args.sort,
        ollama_only=args.ollama_only
    ))

if __name__ == "__main__":
    main()
