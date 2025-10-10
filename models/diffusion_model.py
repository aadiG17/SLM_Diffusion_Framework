# Load base diffusion model (Stable Diffusion, SDXL, or custom fine-tuned)
from diffusers import StableDiffusionPipeline
import torch

def load_diffusion_model(model_name="runwayml/stable-diffusion-v1-5"):
    # Select appropriate device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load the diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    print(f"✅ Diffusion model loaded on: {device.upper()}")
    return pipe
