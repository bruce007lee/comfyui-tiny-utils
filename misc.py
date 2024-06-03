from PIL import Image
from .modules import logger as loggerUtil, imageUtils
import cv2
import numpy as np

class ImageCleanAlpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

    CATEGORY = "TinyUtils"

    def generate(self, mask, trans_info):
        image = imageUtils.tensor2pil(mask)

        im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        output_image = warp_im(im, trans_info[0], trans_info[1])
        output_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        
        output_image = imageUtils.pil2tensor(output_image.convert("L"))
        return (torch.cat([output_image], dim=0),)