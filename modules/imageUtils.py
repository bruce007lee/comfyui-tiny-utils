from PIL import Image, ImageOps
import numpy as np
import torch
from comfy.model_management import get_torch_device
DEVICE = get_torch_device()

def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# Convert to comfy
def pil2comfy(img):
    img = ImageOps.exif_transpose(img)
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image


# Convert PIL to Tensor
# 图片转张量
def pil2tensor(image, device=DEVICE):
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        raise Exception("Input image should be either PIL Image!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
        print(f"Prepare the imput images")
    elif img.ndim == 2:
        img = img[np.newaxis, ...]
        print(f"Prepare the imput masks")

    assert img.ndim == 3

    try:
        img = img.astype(np.float32) / 255
    except:
        img = img.astype(np.float16) / 255
    
    out_image = torch.from_numpy(img).unsqueeze(0).to(device)
    return out_image