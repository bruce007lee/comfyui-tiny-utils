from .modules import logger as loggerUtil, imageUtils, miscUtils
from PIL import Image, ImageOps, ImageSequence, ImageDraw
import cv2
import numpy as np
import torch
import folder_paths
import hashlib
import os
import node_helpers
import math


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


class ImageTransposeAdvance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_overlay": ("IMAGE",),
                "width": (
                    "INT",
                    {"default": 512, "min": -48000, "max": 48000, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": -48000, "max": 48000, "step": 1},
                ),
                "X": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                "Y": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                "rotation": (
                    "FLOAT",
                    {"default": 0, "min": -360, "max": 360, "step": 0.01},
                ),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_transpose"

    CATEGORY = "TinyUtils"

    def image_transpose(
        self,
        image: torch.Tensor,
        image_overlay: torch.Tensor,
        width: int,
        height: int,
        X: int,
        Y: int,
        rotation: int,
        feathering: int = 0,
    ):
        return (
            imageUtils.pil2tensor(
                self.apply_transpose_image(
                    imageUtils.tensor2pil(image),
                    imageUtils.tensor2pil(image_overlay),
                    (width, height),
                    (X, Y),
                    rotation,
                    feathering,
                )
            ),
        )

    def calculate_differ(
        self,
        pos: tuple[int, int],
        oriSize: tuple[int, int],
        newSize: tuple[int, int],
        rotate: int,
    ) -> tuple[int, int]:
        w = oriSize[0]
        h = oriSize[1]

        p = miscUtils.rotatePoint(
            pos, (math.floor(pos[0] + w / 2), math.floor(pos[1] + h / 2)), rotate
        )

        x = 0
        y = math.floor(newSize[1] / 2) + pos[1] - p[1]

        if rotate > 0:
            x = math.floor(newSize[0] / 2) + pos[0] - p[0]
            y = 0

        if rotate > 90 or rotate < -90:
            x = math.floor(newSize[0] / 2) + pos[0] - p[0]
            y = math.floor(newSize[1] / 2) + pos[1] - p[1]

        return (x, y)

    def apply_transpose_image(
        self, image_bg: Image, image_element: Image, size, loc, rotate=0, feathering=0
    ):

        # Apply transformations to the element image
        image_element = image_element.resize(size)
        image_element = image_element.rotate(
            -rotate, expand=True, resample=Image.Resampling.BICUBIC
        )

        imgSize = image_element.size

        # Create a mask for the image with the faded border
        if feathering > 0:
            mask = Image.new("L", imgSize, 255)  # Initialize with 255 instead of 0
            draw = ImageDraw.Draw(mask)
            for i in range(feathering):
                alpha_value = int(
                    255 * (i + 1) / feathering
                )  # Invert the calculation for alpha value
                draw.rectangle(
                    (i, i, imgSize[0] - i, imgSize[1] - i),
                    fill=alpha_value,
                )
            alpha_mask = Image.merge("RGBA", (mask, mask, mask, mask))
            image_element = Image.composite(
                image_element,
                Image.new("RGBA", imgSize, (0, 0, 0, 0)),
                alpha_mask,
            )
        differ = self.calculate_differ(loc, size, imgSize, rotate)
        loc = (loc[0] - differ[0], loc[1] - differ[1])

        # Create a new image of the same size as the base image with an alpha channel
        new_image = Image.new("RGBA", image_bg.size, (0, 0, 0, 0))
        new_image.paste(image_element, loc)

        # Paste the new image onto the base image
        image_bg = image_bg.convert("RGBA")
        image_bg.paste(new_image, (0, 0), new_image)

        return image_bg
