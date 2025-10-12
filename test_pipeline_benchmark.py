import time
import torch
from models.slm_model import load_slm
from models.diffusion_model import load_diffusion_model

def print_memory_usage():
    if torch.backends.mps.is_available():
        mem_alloc = torch.mps.current_allocated_memory() / 1e6
        mem_reserved = torch.mps.driver_allocated_memory() / 1e6
        print(f" MPS Memory: {mem_alloc:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
    elif torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1e6
        mem_reserved = torch.cuda.memory_reserved() / 1e6
        print(f" CUDA Memory: {mem_alloc:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
    else:
        print(" Running on CPU — memory tracking not available.")

def test_pipeline_benchmark():
    # Start total timer
    total_start = time.time()

    print("🔹 Loading Small Language Model...")
    start = time.time()
    tokenizer, slm = load_slm("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f" SLM loaded in {time.time() - start:.2f} seconds.")
    print_memory_usage()

    print("\n🔹 Loading Diffusion Model...")
    start = time.time()
    diffusion = load_diffusion_model("runwayml/stable-diffusion-v1-5")
    print(f" Diffusion model loaded in {time.time() - start:.2f} seconds.")
    print_memory_usage()

    # Text generation
    prompt = "Describe a surreal cyberpunk city glowing under neon lights."
    print("\n Generating text with SLM...")
    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(slm.device)
    outputs = slm.generate(**inputs, max_new_tokens=30)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f" Text generated in {time.time() - start:.2f} seconds.")
    print_memory_usage()

    print("\n Generated Prompt:")
    print(generated_text)

    # Image generation
    print("\n Generating image with Diffusion model...")
    start = time.time()
    image = diffusion(generated_text).images[0]
    image.save("benchmark_output.png")
    print(f" Image saved as 'benchmark_output.png' in {time.time() - start:.2f} seconds.")
    print_memory_usage()

    print(f"\n⏱ Total pipeline runtime: {time.time() - total_start:.2f} seconds.")
    print("\n Benchmark completed successfully.")

if __name__ == "__main__":
    test_pipeline_benchmark()
