import copy
import os
import sys
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

inpa_basedir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models", "sams"
)
# if inpa_basedir not in sys.path:
#     sys.path.append(inpa_basedir)

models_dir = os.path.normpath(inpa_basedir)

print("[DEBUG]models_dir:", models_dir)

from .ia_get_dataset_colormap import create_pascal_label_colormap  # noqa: E402
from .ia_sam_manager import get_sam_mask_generator  # noqa: E402


def sam_file_path(sam_id: str) -> str:
    """Get SAM file path.

    Args:
        sam_id (str): SAM ID

    Returns:
        str: SAM file path
    """
    return os.path.join(models_dir, sam_id)


def sam_file_exists(sam_id: str) -> bool:
    """Check if SAM file exists.

    Args:
        sam_id (str): SAM ID

    Returns:
        bool: True if SAM file exists else False
    """
    sam_checkpoint = sam_file_path(sam_id)

    return os.path.isfile(sam_checkpoint)


def check_inputs_generate_sam_masks(
    input_image: Union[np.ndarray, Image.Image],
    sam_id: str,
    anime_style_chk: bool = False,
) -> None:
    """Check generate SAM masks inputs.

    Args:
        input_image (Union[np.ndarray, Image.Image]): input image
        sam_id (str): SAM ID
        anime_style_chk (bool): anime style check

    Returns:
        None
    """
    if input_image is None or not isinstance(input_image, (np.ndarray, Image.Image)):
        raise ValueError("Invalid input image")

    if sam_id is None or not isinstance(sam_id, str):
        raise ValueError("Invalid SAM ID")

    if anime_style_chk is None or not isinstance(anime_style_chk, bool):
        raise ValueError("Invalid anime style check")


def convert_input_image(input_image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Convert input image.

    Args:
        input_image (Union[np.ndarray, Image.Image]): input image

    Returns:
        np.ndarray: converted input image
    """
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)

    if input_image.ndim == 2:
        input_image = input_image[:, :, np.newaxis]

    if input_image.shape[2] == 1:
        input_image = np.concatenate([input_image] * 3, axis=-1)

    return input_image


def generate_sam_masks(
    input_image: Union[np.ndarray, Image.Image],
    sam_id: str,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    crop_n_layers: int = 0,
    crop_n_points_downscale_factor: int = 1,
    min_mask_region_area: int = 0,
    anime_style_chk: bool = False,
) -> List[Dict[str, Any]]:
    """Generate SAM masks.

    Args:
        input_image (Union[np.ndarray, Image.Image]): input image
        sam_id (str): SAM ID
        anime_style_chk (bool): anime style check

    Returns:
        List[Dict[str, Any]]: SAM masks
    """
    check_inputs_generate_sam_masks(input_image, sam_id, anime_style_chk)
    input_image = convert_input_image(input_image)

    sam_checkpoint = sam_file_path(sam_id)
    sam_mask_generator = get_sam_mask_generator(
        sam_checkpoint,
        points_per_side,
        pred_iou_thresh,
        stability_score_thresh,
        crop_n_layers,
        crop_n_points_downscale_factor,
        min_mask_region_area,
    )
    # ia_logging.info(f"{sam_mask_generator.__class__.__name__} {sam_id}")

    sam_masks = sam_mask_generator.generate(input_image)

    if anime_style_chk:
        for sam_mask in sam_masks:
            sam_mask_seg = sam_mask["segmentation"]
            sam_mask_seg = cv2.morphologyEx(
                sam_mask_seg.astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((5, 5), np.uint8),
            )
            sam_mask_seg = cv2.morphologyEx(
                sam_mask_seg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
            )
            sam_mask["segmentation"] = sam_mask_seg.astype(bool)

    # ia_logging.info("sam_masks: {}".format(len(sam_masks)))

    sam_masks = copy.deepcopy(sam_masks)
    return sam_masks


def sort_masks_by_area(
    sam_masks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sort mask by area.

    Args:
        sam_masks (List[Dict[str, Any]]): SAM masks

    Returns:
        List[Dict[str, Any]]: sorted SAM masks
    """
    return sorted(
        sam_masks, key=lambda x: np.sum(x.get("segmentation").astype(np.uint32))
    )


def get_seg_colormap() -> np.ndarray:
    """Get segmentation colormap.

    Returns:
        np.ndarray: segmentation colormap
    """
    cm_pascal = create_pascal_label_colormap()
    seg_colormap = cm_pascal
    seg_colormap = np.array([c for c in seg_colormap if max(c) >= 64], dtype=np.uint8)

    return seg_colormap


def insert_mask_to_sam_masks(
    sam_masks: List[Dict[str, Any]],
    insert_mask: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Insert mask to SAM masks.

    Args:
        sam_masks (List[Dict[str, Any]]): SAM masks
        insert_mask (Dict[str, Any]): insert mask

    Returns:
        List[Dict[str, Any]]: SAM masks
    """
    if (
        insert_mask is not None
        and isinstance(insert_mask, dict)
        and "segmentation" in insert_mask
    ):
        if (
            len(sam_masks) > 0
            and sam_masks[0]["segmentation"].shape == insert_mask["segmentation"].shape
            and np.any(insert_mask["segmentation"])
        ):
            sam_masks.insert(0, insert_mask)
            # ia_logging.info("insert mask to sam_masks")

    return sam_masks


def create_seg_color_image(
    input_image: Union[np.ndarray, Image.Image],
    sam_masks: List[Dict[str, Any]],
) -> np.ndarray:
    """Create segmentation color image.

    Args:
        input_image (Union[np.ndarray, Image.Image]): input image
        sam_masks (List[Dict[str, Any]]): SAM masks

    Returns:
        np.ndarray: segmentation color image
    """
    input_image = convert_input_image(input_image)

    seg_colormap = get_seg_colormap()
    sam_masks = sam_masks[: len(seg_colormap)]

    with tqdm(total=len(sam_masks), desc="Processing segments") as progress_bar:
        canvas_image = np.zeros((*input_image.shape[:2], 1), dtype=np.uint8)
        for idx, seg_dict in enumerate(sam_masks[0 : min(255, len(sam_masks))]):
            seg_mask = np.expand_dims(
                seg_dict["segmentation"].astype(np.uint8), axis=-1
            )
            canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
            seg_color = np.array([idx + 1], dtype=np.uint8) * seg_mask * canvas_mask
            canvas_image = canvas_image + seg_color
            progress_bar.update(1)
        seg_colormap = np.insert(seg_colormap, 0, [0, 0, 0], axis=0)
        temp_canvas_image = np.apply_along_axis(
            lambda x: seg_colormap[x[0]], axis=-1, arr=canvas_image
        )
        if len(sam_masks) > 255:
            canvas_image = canvas_image.astype(bool).astype(np.uint8)
            for idx, seg_dict in enumerate(sam_masks[255 : min(509, len(sam_masks))]):
                seg_mask = np.expand_dims(
                    seg_dict["segmentation"].astype(np.uint8), axis=-1
                )
                canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
                seg_color = np.array([idx + 2], dtype=np.uint8) * seg_mask * canvas_mask
                canvas_image = canvas_image + seg_color
                progress_bar.update(1)
            seg_colormap = seg_colormap[256:]
            seg_colormap = np.insert(seg_colormap, 0, [0, 0, 0], axis=0)
            seg_colormap = np.insert(seg_colormap, 0, [0, 0, 0], axis=0)
            canvas_image = np.apply_along_axis(
                lambda x: seg_colormap[x[0]], axis=-1, arr=canvas_image
            )
            canvas_image = temp_canvas_image + canvas_image
        else:
            canvas_image = temp_canvas_image
    ret_seg_image = canvas_image.astype(np.uint8)

    return ret_seg_image


# -------------main------------#
sam_dict = dict(
    sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None
)


# @offload_reload_decorator
# @clear_cache_decorator
def run_sam(
    input_image: Image,
    sam_model_id: str,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    crop_n_layers: int = 0,
    crop_n_points_downscale_factor: int = 1,
    min_mask_region_area: int = 0,
):
    global sam_dict
    if not sam_file_exists(sam_model_id):
        raise Exception(f"{sam_model_id} not found, please download")

    if input_image is None:
        raise Exception("Input image not found")

    # set_ia_config(IAConfig.KEYS.SAM_MODEL_ID, sam_model_id, IAConfig.SECTIONS.USER)

    # if sam_dict["sam_masks"] is not None:
    #     sam_dict["sam_masks"] = None
    # gc.collect()

    # ia_logging.info(f"input_image: {input_image.shape} {input_image.dtype}")

    try:
        sam_masks = generate_sam_masks(
            input_image,
            sam_model_id,
            points_per_side,
            pred_iou_thresh,
            stability_score_thresh,
            crop_n_layers,
            crop_n_points_downscale_factor,
            min_mask_region_area,
        )
        sam_masks = sort_masks_by_area(sam_masks)
        sam_masks = insert_mask_to_sam_masks(sam_masks, sam_dict["pad_mask"])

        seg_image = create_seg_color_image(input_image, sam_masks)

        # sam_dict["sam_masks"] = sam_masks

    except Exception as e:
        # print(e)
        # print(traceback.format_exc())
        # ia_logging.error(str(e))
        # ret_sam_image = None if sam_image is None else gr.update()
        raise e

    return Image.fromarray(seg_image)
