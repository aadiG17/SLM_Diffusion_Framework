import time
import torch
from PIL import Image
import torchvision.transforms as T
from models.slm_model import load_slm
from models.diffusion_model import load_diffusion_model

# ---- NEW: CLIP imports ----
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

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

# ---- NEW: Evaluation function ----
def evaluate_clip_metrics(text, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n🔎 Loading CLIP model for evaluation...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(" Computing CLIP embeddings...")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # CLIP Processing
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_emb = outputs.image_embeds      # (1, 512)
        text_emb = outputs.text_embeds        # (1, 512)

    # Normalize embeddings
    image_emb_norm = image_emb / image_emb.norm(dim=-1, keepdim=True)
    text_emb_norm = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # ---- Metric 1: CLIP Similarity ----
    clip_similarity = (image_emb_norm @ text_emb_norm.T).item()

    # ---- Metric 2: L2 Distance ----
    l2_distance = F.pairwise_distance(image_emb, text_emb).item()

    print("\n📊 Evaluation Metrics:")
    print(f"   🔹 CLIP Similarity (S_CLIP): {clip_similarity:.4f}")
    print(f"   🔹 L2 Embedding Distance   : {l2_distance:.4f}")

    return clip_similarity, l2_distance


def test_pipeline_benchmark():
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

    # ---- test prompt ----
    prompt = "A cat wearing astronaut suit on Mars."

    print("\n📝 Generating text with SLM...")
    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(slm.device)
    outputs = slm.generate(**inputs, max_new_tokens=30)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f" Text generated in {time.time() - start:.2f} seconds.")

    print("\n Generated Prompt:")
    print(generated_text)

    # ---- Generate Image ----
    print("\n🎨 Generating image with Diffusion...")
    start = time.time()
    image = diffusion(generated_text).images[0]
    out_path = "generated_output_eval.png"
    image.save(out_path)
    print(f" Image saved: {out_path}")
    print_memory_usage()

    # ---- NEW: Run evaluation ----
    clip_score, l2_metric = evaluate_clip_metrics(generated_text, out_path)

    print("\n⏱ Total pipeline runtime:", time.time() - total_start)
    print("\n Benchmark completed successfully.")
    print("\n Final Metrics:")
    print(f"   ✓ CLIP Similarity: {clip_score:.4f}")
    print(f"   ✓ L2 Distance    : {l2_metric:.4f}")


if __name__ == "__main__":
    test_pipeline_benchmark()
