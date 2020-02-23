# -*- coding: utf-8 -*-
# Copyright (c) ZERO Lab, Inc. and its affiliates. All Rights Reserved

import logging
import torch

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs

from ..utils import multi_apply

logger = logging.getLogger(__name__)


__all__ = ['PanFastRCNNOutputs', 'mask_rcnn_inference']


class PanFastRCNNOutputs(FastRCNNOutputs):

    def inference(self, score_thresh, sog_score_thresh, nms_thresh, topk_per_image):
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        det_instances, _ = fast_rcnn_inference(
            image_shapes, boxes, scores, score_thresh, nms_thresh, topk_per_image
        )

        boxes, scores, classes = self.cls_agnostic_convert(boxes, scores)
        pan_instances, _ = fast_rcnn_inference(
            image_shapes, boxes, scores, sog_score_thresh, nms_thresh, topk_per_image, classes
        )

        return det_instances, pan_instances

    def cls_agnostic_convert(self, boxes, scores):
        ret_boxes = []
        ret_scores = []
        ret_classes = []
        for box, score in zip(boxes, scores):
            score = score[:, :-1]
            num_bboxes, num_classes = score.size()
            score = score.reshape(-1)
            score = torch.stack([score, torch.zeros_like(score)], dim=-1)

            box = box.reshape(-1, 4)

            classes = torch.arange(num_classes, dtype=torch.long, device=score.device)
            classes = classes.repeat(num_bboxes).reshape(-1, 1)

            ret_boxes.append(box)
            ret_scores.append(score)
            ret_classes.append(classes)

        return ret_boxes, ret_scores, ret_classes


def fast_rcnn_inference(
    image_shapes, boxes, scores, score_thresh, nms_thresh, topk_per_image, classes=None
):
    if classes is None:
        result_per_image = multi_apply(
            fast_rcnn_inference_single_image,
            image_shapes,
            boxes,
            scores,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            topk_per_image=topk_per_image
        )
    else:
        result_per_image = multi_apply(
            fast_rcnn_inference_single_image,
            image_shapes,
            boxes,
            scores,
            classes,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            topk_per_image=topk_per_image
        )

    return result_per_image


def fast_rcnn_inference_single_image(
    image_shape, boxes, scores, classes=None, score_thresh=0.05, nms_thresh=0.5, topk_per_image=1000
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    replace_cls = classes is not None
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    if replace_cls:
        classes = classes[filter_mask]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    if replace_cls:
        result.pred_classes = classes[keep]
    else:
        result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    # Select masks corresponding to the predicted classes
    num_masks = pred_mask_logits.shape[0]
    class_pred = cat([i.pred_classes for i in pred_instances])
    indices = torch.arange(num_masks, device=class_pred.device)
    mask_logits_pred = pred_mask_logits[indices, class_pred][:, None]
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_logits_pred = mask_logits_pred.split(num_boxes_per_image, dim=0)

    for logit, instances in zip(mask_logits_pred, pred_instances):
        instances.pred_masks = logit.sigmoid()  # (1, Hmask, Wmask)
        instances.mask_logit = logit
