import logging
import numpy as np
import pycocotools.mask as mask_util
import torch

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    polygons_to_bitmask,
)

def annotations_to_instances(annos, image_size, mask_format="polygon"):
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    pan_ids = torch.tensor([obj["pan_id"] for obj in annos])
    target.pan_id = pan_ids

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        poly_masks = PolygonMasks(segms)
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image_size))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                    segm.ndim
                )
                # mask array
                masks.append(segm)
            else:
                raise ValueError(
                    "Cannot convert segmentation of type '{}' to BitMasks!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict, or a full-image segmentation mask "
                    "as a 2D ndarray.".format(type(segm))
                )
        # torch.from_numpy does not support array with negative stride.
        bit_masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        if mask_format == "polygon":
            target.gt_masks = poly_masks
            target.bit_masks = bit_masks
        else:
            target.gt_masks = bit_masks
            target.poly_masks = poly_masks

    return target


def get_relation_gt(instances, sample_num=-1):
    gt_masks = instances.bit_masks.tensor

    num_ins = gt_masks.size(0)
    gt_masks = (gt_masks == 1).float()

    inter_area = (gt_masks[:, None, :, :] * gt_masks).view(num_ins, num_ins, -1).sum(dim=-1)
    mask_area = gt_masks.view(num_ins, -1).sum(dim=-1)
    min_area = torch.min(mask_area[:, None], mask_area)
    intersection_ratio = inter_area.float() / min_area.float()
    intersection_ratio -= torch.eye(num_ins).float()
    relation_mat = (intersection_ratio >= 0.1).float()

    if sample_num != 0:
        indicator = relation_mat.sum(dim=1)
        if sample_num == -1:
            sample_num = (indicator > 0).sum()
        if sample_num == 0:
            return instances, -1 * torch.ones((1, 1))
        order = (-indicator).argsort()
        relation_mat = relation_mat[order][:, order][ :sample_num, :sample_num]
        instances = instances[order]

    return instances, relation_mat


def pan_id2channel_id(pan_seg_gt, pan_ids, ch_id_shift=0, ignore_index=255):
    pan_ids = pan_ids.numpy()
    panel = np.zeros_like(pan_seg_gt)
    panel[panel < ch_id_shift] = 1
    for i, pan_id in enumerate(pan_ids):
        if pan_id == -1:
            continue
        mask = pan_seg_gt == pan_id
        panel[mask] = 1
        pan_seg_gt[mask] = i + ch_id_shift
    pan_seg_gt[panel == 0] = ignore_index

    return pan_seg_gt

