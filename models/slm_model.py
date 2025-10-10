# Define or load Small Language Model (SLM) here
# Example: use TinyLlama, Phi-1.5, or DistilGPT2 for lightweight text understanding

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_slm(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    # Choose device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32)
    model.to(device)

    print(f"✅ SLM loaded on: {device.upper()}")
    return tokenizer, model
