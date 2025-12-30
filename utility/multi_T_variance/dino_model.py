# dino_model.py
import torch
from functools import lru_cache
from transformers import AutoImageProcessor, AutoModel


DEFAULT_MODEL_NAME = "facebook/dinov2-base"


@lru_cache(maxsize=1)
def load_dino(model_name: str = DEFAULT_MODEL_NAME):
    """
    Load DINOv2 model and processor ONCE.
    Subsequent calls reuse the same model (no extra GPU memory / time).

    Returns:
        model      : DINOv2 torch.nn.Module (eval mode, on device)
        processor  : AutoImageProcessor
        device     : torch.device
    """
    print(f"[DINO] Loading model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"[DINO] Model loaded on device: {device}")
    return model, processor, device
