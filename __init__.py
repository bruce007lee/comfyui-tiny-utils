from .face import FaceAlign, FaceAlignImageProcess, FaceAlignMaskProcess
from .misc import ImageFillColorByMask, CropImageByMask

NODE_CLASS_MAPPINGS = {
    "FaceAlign": FaceAlign,
    "FaceAlignImageProcess": FaceAlignImageProcess,
    "FaceAlignMaskProcess": FaceAlignMaskProcess,
    "ImageFillColorByMask": ImageFillColorByMask,
    "CropImageByMask": CropImageByMask,
}
