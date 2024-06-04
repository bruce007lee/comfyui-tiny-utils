from PIL import Image, ImageColor
from .modules import logger as loggerUtil, imageUtils
import cv2
import numpy as np
import torch


class ImageFillColorByMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color": ("STRING", {"default": "#ffffff"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

    CATEGORY = "TinyUtils"

    def generate(self, image, mask, color="#ffffff"):
        image = imageUtils.tensor2pil(image)
        mask = imageUtils.tensor2pil(mask)

        output_image = imageUtils.fillColorByMask(
            image, mask, ImageColor.getcolor(color, "RGBA")
        )
        output_image = imageUtils.pil2comfy(output_image)
        return (torch.cat([output_image], dim=0),)