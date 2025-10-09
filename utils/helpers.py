import os
from PIL import Image

def load_image(path):
    return Image.open(path)

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    print(f" Saved image at {path}")
