from models.slm_model import load_slm
from models.diffusion_model import load_diffusion_model
import torch

def test_pipeline():
    # Load models
    print("🔹 Loading Small Language Model...")
    tokenizer, slm = load_slm("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print("🔹 Loading Diffusion Model...")
    diffusion = load_diffusion_model("runwayml/stable-diffusion-v1-5")

    # Generate a short text with SLM
    prompt = "Describe a serene fantasy landscape for image generation."
    inputs = tokenizer(prompt, return_tensors="pt").to(slm.device)
    outputs = slm.generate(**inputs, max_new_tokens=30)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n📝 Generated Prompt from SLM:")
    print(generated_text)

    # Generate an image using the Diffusion model
    print("\n🎨 Generating image with Diffusion model...")
    image = diffusion(generated_text).images[0]
    image.save("test_output.png")
    print("✅ Image saved as 'test_output.png'")

if __name__ == "__main__":
    test_pipeline()
