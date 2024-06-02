from PIL import Image
from .modules import devices, logger as loggerUtil, imageUtils
import platform
import os
import cv2
import numpy as np
import torch

logger = loggerUtil.logger


class Cleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

    CATEGORY = "Cleaner"

    def generate(self, image1, image2):
        image = imageUtils.tensor2pil(image)
        mask = imageUtils.tensor2pil(mask)

        output_image = Image.fromarray(output_image)

        output_image = pil2comfy(output_image)
        del model
        return (torch.cat([output_image], dim=0),)
