from PIL import Image, ImageOps, ImageColor
from typing import Union, List
import numpy as np
import torch
from comfy.model_management import get_torch_device

DEVICE = get_torch_device()


def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
    if len(tensor.shape) == 3:  # Single image
        return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    else:  # Batch of images
        return [
            np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor
        ]


def tensor_mask2image(mask: torch.Tensor) -> torch.Tensor:
    result = (
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        .movedim(1, -1)
        .expand(-1, -1, -1, 3)
    )
    return result


def tensor_image2mask(image: torch.Tensor) -> torch.Tensor:
    mask = image[:, :, :, 0]
    return mask


# Convert to comfy
def pil2comfy(img, mode="RGB") -> torch.Tensor:
    img = ImageOps.exif_transpose(img)
    image = img.convert(mode)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image


# Convert PIL to Tensor
# 图片转张量
def pil2tensor(image, device=DEVICE) -> torch.Tensor:
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


def fillColorByMask(image: Image, mask: Image, color) -> Image:
    if mask.mode != "RGB":
        mask = mask.convert("RGB")
    color = ImageColor.getcolor(color, "RGBA")
    maskDatas = mask.getdata()
    datas = image.getdata()
    new_datas = []
    index = 0
    for item in datas:
        md = maskDatas[index]
        if md[0] != 0:
            new_datas.append(color)
        else:
            new_datas.append(item)
        index += 1
    img = image.copy()
    img.putdata(new_datas)
    return img


def cropImageByMask(image: Image, mask: Image, color="#ffffff") -> Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if mask.mode != "RGB":
        mask = mask.convert("RGB")

    if image.size[0] != mask.size[0] or image.size[1] != mask.size[1]:
        raise Exception("Image size not match mask size")

    color = ImageColor.getcolor(color, "RGBA")
    maskDatas = mask.getdata()
    datas = image.getdata()
    new_datas = []
    index = 0
    for item in datas:
        md = maskDatas[index]
        if md[0] != 0:
            new_datas.append(item)
        else:
            new_datas.append(color)
        index += 1
    img = image.copy()
    img.putdata(new_datas)
    return img
