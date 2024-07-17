from PIL import Image, ImageColor
from .modules import logger as loggerUtil, imageUtils
from PIL import Image, ImageOps, ImageSequence
import cv2
import numpy as np
import torch
import folder_paths
import hashlib
import os
import node_helpers


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

        output_image = imageUtils.fillColorByMask(image, mask, color)
        output_image = imageUtils.pil2comfy(output_image)
        return (torch.cat([output_image], dim=0),)


class CropImageByMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color": ("STRING", {"default": "#ffffff"}),
                "mode": (["RGB", "RGBA"], {"default": "RGB"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

    CATEGORY = "TinyUtils"

    def generate(self, image, mask, color="#ffffff", mode="RGB"):
        image = imageUtils.tensor2pil(image)
        mask = imageUtils.tensor2pil(mask)

        output_image = imageUtils.cropImageByMask(image, mask, color)
        output_image = imageUtils.pil2comfy(output_image, mode)
        return (torch.cat([output_image], dim=0),)


class LoadImageAdvance:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "mode": (["RGB", "RGBA"], {"default": "RGB"}),
            },
        }

    CATEGORY = "TinyUtils"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_adv"

    def load_image_adv(self, image, mode="RGB"):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))

            if mode == "RGBA":
                image = i.convert("RGBA")
            else:
                image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
