import os
import pandas as pd
from datasets import load_dataset
from PIL import Image
import torch
from torchvision import transforms

def prepare_instruction_image_dataset(
    hf_dataset_name: str = "data-is-better-together/open-image-preferences-v1-binarized",
    out_csv: str = "data/instruction_image_pairs.csv",
    out_image_dir: str = "data/images/",
    max_samples: int = None
):
    """
    Loads the Hugging Face dataset of image preferences,
    selects the preferred image for each prompt,
    saves them in our format: instruction_text → target_image.
    """
    ds = load_dataset(hf_dataset_name, split="train")  # :contentReference[oaicite:0]{index=0}
    records = []
    os.makedirs(out_image_dir, exist_ok=True)

    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        prompt = row["prompt"]
        chosen_img_url = row["chosen"]  # URL or path
        # Optionally download the image if URL, or copy if path.
        # Here you assume “chosen” is a local path or accessible URL
        # For simplicity: we skip download logic.
        img_name = f"img_{i}.png"
        img_path = os.path.join(out_image_dir, img_name)

        # Instead of download, you may skip and link to existing file.
        # For now we just record reference
        records.append({"instruction": prompt, "image_path": img_path})

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved instruction-image CSV: {out_csv} with {len(records)} rows")
    print(f"Images directory: {out_image_dir} (you need to ensure images are available)")
    return df

if __name__ == "__main__":
    df_pairs = prepare_instruction_image_dataset(max_samples=1000)
