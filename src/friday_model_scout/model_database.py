"""Curated database of LLM models for scouting."""

from typing import List, Dict, Any

# Model data schema:
# name: str
# params: str (e.g., "8B")
# vram_req: Dict[str, float] (quantization: VRAM in GB)
# ram_req: float (Recommended RAM in GB if no GPU)
# tags: List[str]
# ollama_name: Optional[str]

MODELS: List[Dict[str, Any]] = [
    # --- Meta ---
    {
        "name": "Llama 3.1 8B",
        "params": "8B",
        "vram_req": {"Q4_K_M": 5.5, "Q5_K_M": 6.5, "Q8_0": 9.0},
        "ram_req": 8,
        "tags": ["meta", "general", "popular"],
        "ollama_name": "llama3.1:8b"
    },
    {
        "name": "Llama 3.1 70B",
        "params": "70B",
        "vram_req": {"Q4_K_M": 43.0, "Q5_K_M": 49.0, "Q8_0": 75.0},
        "ram_req": 64,
        "tags": ["meta", "large", "popular"],
        "ollama_name": "llama3.1:70b"
    },
    {
        "name": "Llama 3.1 405B",
        "params": "405B",
        "vram_req": {"Q4_K_M": 230.0, "Q5_K_M": 270.0, "Q8_0": 410.0},
        "ram_req": 512,
        "tags": ["meta", "large", "frontier"],
        "ollama_name": "llama3.1:405b"
    },
    {
        "name": "CodeLlama 7B",
        "params": "7B",
        "vram_req": {"Q4_K_M": 4.5, "Q5_K_M": 5.5, "Q8_0": 8.0},
        "ram_req": 8,
        "tags": ["meta", "coding"],
        "ollama_name": "codellama:7b"
    },
    {
        "name": "CodeLlama 13B",
        "params": "13B",
        "vram_req": {"Q4_K_M": 8.5, "Q5_K_M": 10.0, "Q8_0": 14.0},
        "ram_req": 16,
        "tags": ["meta", "coding"],
        "ollama_name": "codellama:13b"
    },
    {
        "name": "CodeLlama 34B",
        "params": "34B",
        "vram_req": {"Q4_K_M": 20.0, "Q5_K_M": 24.0, "Q8_0": 36.0},
        "ram_req": 32,
        "tags": ["meta", "coding"],
        "ollama_name": "codellama:34b"
    },
    {
        "name": "CodeLlama 70B",
        "params": "70B",
        "vram_req": {"Q4_K_M": 40.0, "Q5_K_M": 48.0, "Q8_0": 75.0},
        "ram_req": 64,
        "tags": ["meta", "coding"],
        "ollama_name": "codellama:70b"
    },

    # --- Mistral AI ---
    {
        "name": "Mistral 7B v0.3",
        "params": "7B",
        "vram_req": {"Q4_K_M": 4.5, "Q5_K_M": 5.5, "Q8_0": 8.0},
        "ram_req": 8,
        "tags": ["mistral", "general"],
        "ollama_name": "mistral"
    },
    {
        "name": "Mistral Nemo 12B",
        "params": "12B",
        "vram_req": {"Q4_K_M": 8.5, "Q5_K_M": 10.0, "Q8_0": 13.5},
        "ram_req": 16,
        "tags": ["mistral", "general"],
        "ollama_name": "mistral-nemo"
    },
    {
        "name": "Mistral Large 2",
        "params": "123B",
        "vram_req": {"Q4_K_M": 70.0, "Q5_K_M": 85.0, "Q8_0": 130.0},
        "ram_req": 128,
        "tags": ["mistral", "large"],
        "ollama_name": "mistral-large"
    },
    {
        "name": "Mixtral 8x7B",
        "params": "47B",
        "vram_req": {"Q4_K_M": 26.0, "Q5_K_M": 30.0, "Q8_0": 48.0},
        "ram_req": 32,
        "tags": ["mistral", "moe"],
        "ollama_name": "mixtral"
    },
    {
        "name": "Mixtral 8x22B",
        "params": "141B",
        "vram_req": {"Q4_K_M": 85.0, "Q5_K_M": 100.0, "Q8_0": 160.0},
        "ram_req": 128,
        "tags": ["mistral", "moe", "large"],
        "ollama_name": "mixtral:8x22b"
    },

    # --- Google ---
    {
        "name": "Gemma 2 2B",
        "params": "2B",
        "vram_req": {"Q4_K_M": 2.0, "Q5_K_M": 2.5, "Q8_0": 3.5},
        "ram_req": 4,
        "tags": ["google", "small", "fast"],
        "ollama_name": "gemma2:2b"
    },
    {
        "name": "Gemma 2 9B",
        "params": "9B",
        "vram_req": {"Q4_K_M": 6.0, "Q5_K_M": 7.5, "Q8_0": 10.5},
        "ram_req": 16,
        "tags": ["google", "general"],
        "ollama_name": "gemma2:9b"
    },
    {
        "name": "Gemma 2 27B",
        "params": "27B",
        "vram_req": {"Q4_K_M": 17.0, "Q5_K_M": 20.0, "Q8_0": 30.0},
        "ram_req": 32,
        "tags": ["google", "general"],
        "ollama_name": "gemma2:27b"
    },

    # --- Microsoft ---
    {
        "name": "Phi-3.5 Mini",
        "params": "3.8B",
        "vram_req": {"Q4_K_M": 2.8, "Q5_K_M": 3.5, "Q8_0": 4.5},
        "ram_req": 4,
        "tags": ["microsoft", "small", "fast"],
        "ollama_name": "phi3.5"
    },
    {
        "name": "Phi-3.5 MoE",
        "params": "42B",
        "vram_req": {"Q4_K_M": 24.0, "Q5_K_M": 28.0, "Q8_0": 44.0},
        "ram_req": 32,
        "tags": ["microsoft", "moe"],
        "ollama_name": "phi3.5:moe"
    },
    {
        "name": "Phi-3 Medium",
        "params": "14B",
        "vram_req": {"Q4_K_M": 9.5, "Q5_K_M": 11.0, "Q8_0": 15.0},
        "ram_req": 16,
        "tags": ["microsoft", "general"],
        "ollama_name": "phi3:medium"
    },

    # --- Alibaba (Qwen) ---
    {
        "name": "Qwen 2.5 0.5B",
        "params": "0.5B",
        "vram_req": {"Q4_K_M": 0.5, "Q5_K_M": 0.6, "Q8_0": 0.9},
        "ram_req": 1,
        "tags": ["alibaba", "small", "edge"],
        "ollama_name": "qwen2.5:0.5b"
    },
    {
        "name": "Qwen 2.5 1.5B",
        "params": "1.5B",
        "vram_req": {"Q4_K_M": 1.2, "Q5_K_M": 1.5, "Q8_0": 2.2},
        "ram_req": 2,
        "tags": ["alibaba", "small"],
        "ollama_name": "qwen2.5:1.5b"
    },
    {
        "name": "Qwen 2.5 3B",
        "params": "3B",
        "vram_req": {"Q4_K_M": 2.2, "Q5_K_M": 2.6, "Q8_0": 3.8},
        "ram_req": 4,
        "tags": ["alibaba", "small"],
        "ollama_name": "qwen2.5:3b"
    },
    {
        "name": "Qwen 2.5 7B",
        "params": "7B",
        "vram_req": {"Q4_K_M": 4.8, "Q5_K_M": 5.8, "Q8_0": 8.5},
        "ram_req": 8,
        "tags": ["alibaba", "general"],
        "ollama_name": "qwen2.5:7b"
    },
    {
        "name": "Qwen 2.5 14B",
        "params": "14B",
        "vram_req": {"Q4_K_M": 9.5, "Q5_K_M": 11.5, "Q8_0": 16.5},
        "ram_req": 16,
        "tags": ["alibaba", "general"],
        "ollama_name": "qwen2.5:14b"
    },
    {
        "name": "Qwen 2.5 32B",
        "params": "32B",
        "vram_req": {"Q4_K_M": 20.0, "Q5_K_M": 24.0, "Q8_0": 36.0},
        "ram_req": 32,
        "tags": ["alibaba", "general"],
        "ollama_name": "qwen2.5:32b"
    },
    {
        "name": "Qwen 2.5 72B",
        "params": "72B",
        "vram_req": {"Q4_K_M": 45.0, "Q5_K_M": 55.0, "Q8_0": 80.0},
        "ram_req": 64,
        "tags": ["alibaba", "large"],
        "ollama_name": "qwen2.5:72b"
    },
    {
        "name": "Qwen 2.5 Coder 7B",
        "params": "7B",
        "vram_req": {"Q4_K_M": 4.8, "Q5_K_M": 5.8, "Q8_0": 8.5},
        "ram_req": 8,
        "tags": ["alibaba", "coding"],
        "ollama_name": "qwen2.5-coder:7b"
    },
    {
        "name": "Qwen 2.5 Coder 32B",
        "params": "32B",
        "vram_req": {"Q4_K_M": 20.0, "Q5_K_M": 24.0, "Q8_0": 36.0},
        "ram_req": 32,
        "tags": ["alibaba", "coding"],
        "ollama_name": "qwen2.5-coder:32b"
    },

    # --- DeepSeek ---
    {
        "name": "DeepSeek Coder V2 Lite",
        "params": "16B",
        "vram_req": {"Q4_K_M": 10.0, "Q5_K_M": 12.0, "Q8_0": 18.0},
        "ram_req": 24,
        "tags": ["deepseek", "coding", "moe"],
        "ollama_name": "deepseek-coder-v2:lite"
    },
    {
        "name": "DeepSeek V2.5",
        "params": "236B",
        "vram_req": {"Q4_K_M": 140.0, "Q5_K_M": 170.0, "Q8_0": 260.0},
        "ram_req": 256,
        "tags": ["deepseek", "large", "moe"],
        "ollama_name": "deepseek-v2.5"
    },
    {
        "name": "DeepSeek Math 7B",
        "params": "7B",
        "vram_req": {"Q4_K_M": 4.8, "Q5_K_M": 5.8, "Q8_0": 8.5},
        "ram_req": 8,
        "tags": ["deepseek", "math"],
        "ollama_name": "deepseek-math"
    },

    # --- Other Notable Models ---
    {
        "name": "Command R",
        "params": "35B",
        "vram_req": {"Q4_K_M": 20.0, "Q5_K_M": 24.0, "Q8_0": 38.0},
        "ram_req": 32,
        "tags": ["cohere", "raging", "tools"],
        "ollama_name": "command-r"
    },
    {
        "name": "Hermes 3 Llama 3.1 8B",
        "params": "8B",
        "vram_req": {"Q4_K_M": 5.5, "Q5_K_M": 6.5, "Q8_0": 9.0},
        "ram_req": 8,
        "tags": ["nous", "general", "instruct"],
        "ollama_name": "hermes3"
    },
    {
        "name": "StarCoder 2 7B",
        "params": "7B",
        "vram_req": {"Q4_K_M": 4.5, "Q5_K_M": 5.5, "Q8_0": 8.0},
        "ram_req": 8,
        "tags": ["bigcode", "coding"],
        "ollama_name": "starcoder2:7b"
    },
    {
        "name": "StableLM 2 1.6B",
        "params": "1.6B",
        "vram_req": {"Q4_K_M": 1.2, "Q5_K_M": 1.5, "Q8_0": 2.2},
        "ram_req": 4,
        "tags": ["stability", "small"],
        "ollama_name": "stable-code:3b" # Approximate
    },
    {
        "name": "StableLM Zephyr 3B",
        "params": "3B",
        "vram_req": {"Q4_K_M": 2.2, "Q5_K_M": 2.6, "Q8_0": 3.8},
        "ram_req": 8,
        "tags": ["stability", "small"],
        "ollama_name": "stablelm-zephyr"
    },
    {
        "name": "TinyLlama 1.1B",
        "params": "1.1B",
        "vram_req": {"Q4_K_M": 0.8, "Q5_K_M": 1.0, "Q8_0": 1.5},
        "ram_req": 2,
        "tags": ["small", "fast"],
        "ollama_name": "tinyllama"
    },
    {
        "name": "Falcon 7B",
        "params": "7B",
        "vram_req": {"Q4_K_M": 4.8, "Q5_K_M": 5.8, "Q8_0": 8.5},
        "ram_req": 8,
        "tags": ["tii", "general"],
        "ollama_name": "falcon:7b"
    },
    {
        "name": "Falcon 40B",
        "params": "40B",
        "vram_req": {"Q4_K_M": 24.0, "Q5_K_M": 28.0, "Q8_0": 44.0},
        "ram_req": 32,
        "tags": ["tii", "general"],
        "ollama_name": "falcon:40b"
    },
    {
        "name": "InternLM 2.5 7B",
        "params": "7B",
        "vram_req": {"Q4_K_M": 4.8, "Q5_K_M": 5.8, "Q8_0": 8.5},
        "ram_req": 8,
        "tags": ["internlm", "general"],
        "ollama_name": "internlm2:7b"
    },
    {
        "name": "InternLM 2.5 20B",
        "params": "20B",
        "vram_req": {"Q4_K_M": 13.0, "Q5_K_M": 16.0, "Q8_0": 24.0},
        "ram_req": 24,
        "tags": ["internlm", "general"],
        "ollama_name": "internlm2:20b"
    },
    {
        "name": "Yi 1.5 6B",
        "params": "6B",
        "vram_req": {"Q4_K_M": 4.0, "Q5_K_M": 4.8, "Q8_0": 7.0},
        "ram_req": 8,
        "tags": ["01ai", "general"],
        "ollama_name": "yi:6b"
    },
    {
        "name": "Yi 1.5 9B",
        "params": "9B",
        "vram_req": {"Q4_K_M": 6.0, "Q5_K_M": 7.5, "Q8_0": 10.5},
        "ram_req": 16,
        "tags": ["01ai", "general"],
        "ollama_name": "yi:9b"
    },
    {
        "name": "Yi 1.5 34B",
        "params": "34B",
        "vram_req": {"Q4_K_M": 20.0, "Q5_K_M": 24.0, "Q8_0": 36.0},
        "ram_req": 32,
        "tags": ["01ai", "general"],
        "ollama_name": "yi:34b"
    },
    {
        "name": "OpenCoder 8B",
        "params": "8B",
        "vram_req": {"Q4_K_M": 5.5, "Q5_K_M": 6.5, "Q8_0": 9.0},
        "ram_req": 8,
        "tags": ["opencoder", "coding"],
        "ollama_name": "opencoder:8b"
    },
    {
        "name": "OpenCoder 1.5B",
        "params": "1.5B",
        "vram_req": {"Q4_K_M": 1.2, "Q5_K_M": 1.5, "Q8_0": 2.2},
        "ram_req": 2,
        "tags": ["opencoder", "coding", "small"],
        "ollama_name": "opencoder:1.5b"
    },
    {
        "name": "H2O Danube 3 4B",
        "params": "4B",
        "vram_req": {"Q4_K_M": 2.8, "Q5_K_M": 3.5, "Q8_0": 4.5},
        "ram_req": 8,
        "tags": ["h2o", "small"],
        "ollama_name": "h2o-danube3:4b"
    },
    {
        "name": "Smaug 72B",
        "params": "72B",
        "vram_req": {"Q4_K_M": 45.0, "Q5_K_M": 55.0, "Q8_0": 80.0},
        "ram_req": 64,
        "tags": ["smaug", "large"],
        "ollama_name": "smaug"
    },
    {
        "name": "Solar 10.7B",
        "params": "10.7B",
        "vram_req": {"Q4_K_M": 7.5, "Q5_K_M": 9.0, "Q8_0": 13.0},
        "ram_req": 16,
        "tags": ["upstage", "general"],
        "ollama_name": "solar"
    },
    {
        "name": "Grok-1",
        "params": "314B",
        "vram_req": {"Q4_K_M": 180.0, "Q5_K_M": 210.0, "Q8_0": 320.0},
        "ram_req": 512,
        "tags": ["xai", "large", "moe"],
        "ollama_name": "grok-1"
    }
]

def get_models() -> List[Dict[str, Any]]:
    return MODELS
