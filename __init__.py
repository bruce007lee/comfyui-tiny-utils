from .face import FaceAlign, FaceAlignImageProcess, FaceAlignMaskProcess
from .image import ImageFillColorByMask, CropImageByMask, LoadImageAdvance

NODE_CLASS_MAPPINGS = {
    "FaceAlign": FaceAlign,
    "FaceAlignImageProcess": FaceAlignImageProcess,
    "FaceAlignMaskProcess": FaceAlignMaskProcess,
    "ImageFillColorByMask": ImageFillColorByMask,
    "CropImageByMask": CropImageByMask,
    "LoadImageAdvance": LoadImageAdvance
}
