import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os

device = "cuda"

# -------------------------------
# 1. Load Base Stable Diffusion
# -------------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)

# -------------------------------
# Load CLIP for scoring
# -------------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def compute_clip_score(prompt, image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        text=[prompt], images=image, return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        res = clip_model(**inputs)

    logits = res.logits_per_text  # similarity
    clip_score = logits.item()

    image_embed = res.image_embeds[0].cpu().numpy()
    text_embed = res.text_embeds[0].cpu().numpy()
    l2_dist = np.linalg.norm(image_embed - text_embed)

    return clip_score, l2_dist


# -------------------------------
# 2. Generate Before-LoRA Image
# -------------------------------
prompt = "A dreamy sunset over a calm lake with pinkish clouds"

print("\n=== Generating BEFORE LoRA Image ===")
before_img = pipe(prompt, num_inference_steps=30).images[0]
os.makedirs("outputs", exist_ok=True)
before_img.save("outputs/before_lora.png")

# -------------------------------
# 3. Load tiny LoRA weights
# -------------------------------
print("\n=== Loading LoRA ===")
sd = torch.load("adapters/tiny_lora/unet_lora.pth")

with torch.no_grad():
    for name, module in pipe.unet.named_modules():
        if isinstance(module, torch.nn.Linear) and name in sd:
            A, B = sd[name]
            module.weight += (B @ A)

# -------------------------------
# 4. Generate After-LoRA Image
# -------------------------------
print("\n=== Generating AFTER LoRA Image ===")
after_img = pipe(prompt, num_inference_steps=30).images[0]
after_img.save("outputs/after_lora.png")

# -------------------------------
# 5. Compute metrics
# -------------------------------
print("\n=== Computing Metrics ===")
before_clip, before_l2 = compute_clip_score(prompt, "outputs/before_lora.png")
after_clip, after_l2 = compute_clip_score(prompt, "outputs/after_lora.png")

print("\n================ Final Results ================")
print(f"Prompt: {prompt}\n")
print(f"CLIP Score BEFORE: {before_clip:.4f}")
print(f"CLIP Score AFTER : {after_clip:.4f}")
print(f"Improvement: {after_clip - before_clip:.4f}\n")

print(f"L2 Distance BEFORE: {before_l2:.4f}")
print(f"L2 Distance AFTER : {after_l2:.4f}")
print(f"Improvement: {before_l2 - after_l2:.4f}")
print("================================================")
