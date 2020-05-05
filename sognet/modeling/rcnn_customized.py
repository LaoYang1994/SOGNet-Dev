# -*- coding: utf-8 -*-
# Copyright (c) ZERO Lab, Inc. and its affiliates. All Rights Reserved

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import batched_nms, cat, Linear, ShapeSpec
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.modeling.box_regression import Box2BoxTransform, apply_deltas_broadcast

from ..utils import multi_apply

logger = logging.getLogger(__name__)


__all__ = ['FastRCNNOutputLayers', 'mask_rcnn_inference']


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


class FastRCNNOutputLayers(nn.Module):
    
    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        test_score_thresh=0.0,
        test_sog_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss.
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatbility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_sog_score_thresh = test_sog_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_sog_score_thresh" : cfg.MODEL.SOGNET.POSTPROCESS.INSTANCES_CONFIDENCE_THRESH,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE
            # fmt: on
        }

    def forward(self, x):
        """
        Returns:
            Tensor: Nx(K+1) scores for each box
            Tensor: Nx4 or Nx(Kx4) bounding box regression deltas.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        return FastRCNNOutputs(
            self.box2box_transform, scores, proposal_deltas, proposals, self.smooth_l1_beta
        ).losses()

    def inference(self, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        scores, proposal_deltas = predictions
        boxes = self.predict_boxes(proposal_deltas, proposals)
        probs = self.predict_probs(scores, proposals)
        image_shapes = [x.image_size for x in proposals]

        det_instances, _ = fast_rcnn_inference(
            image_shapes, boxes, probs,
            self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image
        )

        boxes, probs , classes = self.cls_agnostic_convert(boxes, probs)
        pan_instances, _ = fast_rcnn_inference(
            image_shapes, boxes, probs,
            self.test_sog_score_thresh, self.test_nms_thresh, self.test_topk_per_image, classes
        )

        return det_instances, pan_instances

    def predict_boxes(self, proposal_deltas, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = apply_deltas_broadcast(
            self.box2box_transform, proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, scores, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

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
