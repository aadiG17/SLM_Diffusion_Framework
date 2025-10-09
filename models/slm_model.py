# Define or load Small Language Model (SLM) here
# Example: use TinyLlama, Phi-1.5, or DistilGPT2 for lightweight text understanding

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_slm(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model
