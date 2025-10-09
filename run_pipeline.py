from models.slm_model import load_slm
from models.diffusion_model import load_diffusion_model
from utils.config import *

def main():
    print(" Loading Small Language Model...")
    tokenizer, slm = load_slm(SLM_MODEL_NAME)

    print(" Loading Diffusion Model...")
    diffusion = load_diffusion_model(DIFFUSION_MODEL_NAME)

    print(" Models loaded successfully. Ready for integration and testing.")

if __name__ == "__main__":
    main()
