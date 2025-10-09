# Load base diffusion model (Stable Diffusion, SDXL, or custom fine-tuned)
from diffusers import StableDiffusionPipeline
import torch

def load_diffusion_model(model_name="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe
