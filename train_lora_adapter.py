import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm

# ---------------------------------------------------------
# Simple dummy dataset (your real dataset can be plugged in)
# ---------------------------------------------------------
class SimpleImageTextDataset(Dataset):
    def __init__(self):
        self.data = [
            ("a sunset over a lake", "samples/sunset.jpg"),
            ("a cat sitting on a sofa", "samples/cat.jpg")
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt, path = self.data[idx]
        image = Image.open(path).convert("RGB").resize((512, 512))
        return prompt, image

# ---------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------
def train_lora(output_dir="adapters/tinyllama_magicbrush_lora", rank=4, steps=50):

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")

    unet = pipe.unet

    # ----------------------------------
    # Create LoRA weights manually
    # ----------------------------------
    print("\n💡 Injecting LoRA layers manually...")
    lora_layers = {}

    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Create LoRA matrices
            in_f = module.in_features
            out_f = module.out_features
            
            lora_A = torch.nn.Parameter(torch.zeros((rank, in_f), device="cuda"))
            lora_B = torch.nn.Parameter(torch.zeros((out_f, rank), device="cuda"))

            torch.nn.init.normal_(lora_A, std=0.01)
            torch.nn.init.zeros_(lora_B)

            lora_layers[name] = (lora_A, lora_B)

    print(f"Injected {len(lora_layers)} LoRA layers.")

    # Make LoRA params trainable
    params = []
    for A, B in lora_layers.values():
        params.extend([A, B])

    optimizer = torch.optim.Adam(params, lr=1e-4)

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    dataset = SimpleImageTextDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for step in tqdm(range(steps), desc="Training LoRA"):
        for prompt, image in loader:

            # Encode text
            text_emb = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to("cuda")

            text_context = pipe.text_encoder(text_emb)[0]

            # Sample random noise
            latents = torch.randn((1, 4, 64, 64), device="cuda")
            t = torch.randint(0, 1000, (1,), device="cuda")

            # Forward through UNet
            noise_pred = unet(latents, t, encoder_hidden_states=text_context).sample

            loss = noise_pred.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            break  # Only 1 batch per step

        if step % 10 == 0:
            print("Step:", step, "Loss:", float(loss))

    # ---------------------------------------------------------
    # Save LoRA
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    torch.save(lora_layers, f"{output_dir}/unet_lora.pth")

    print(f"\n Saved LoRA adapter to {output_dir}")

# ---------------------------------------------------------
if __name__ == "__main__":
    train_lora()
