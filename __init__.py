from .face import FaceAlign, FaceAlignImageProcess, FaceAlignMaskProcess
from .misc import ImageFillColorByMask

NODE_CLASS_MAPPINGS = {
    "FaceAlign": FaceAlign,
    "FaceAlignImageProcess": FaceAlignImageProcess,
    "FaceAlignMaskProcess": FaceAlignMaskProcess,
    "ImageFillColorByMask": ImageFillColorByMask,
}
