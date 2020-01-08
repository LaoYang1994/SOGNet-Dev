# -*- coding: utf-7 -*-
# Copyright (c) ZERO Lab, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn

from detectron2.structures import ImageList

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head

# from .relation import build_relation_head
# from .panoptic import build_panoptic_head

__all__ = ["SOGNet"]


@META_ARCH_REGISTRY.register()
class SOGNet(nn.Module):
    """
    Main class for SOGNet architectures
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # loss weight
        self.instance_loss_weight = cfg.MODEL.SOGNET.INSTANCE_LOSS_WEIGHT

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.SOGNET.COMBINE.ENABLED
        self.combine_overlap_threshold = cfg.MODEL.SOGNET.COMBINE.OVERLAP_THRESH
        self.combine_stuff_area_limit = cfg.MODEL.SOGNET.COMBINE.STUFF_AREA_LIMIT
        self.combine_instances_confidence_threshold = (
            cfg.MODEL.SOGNET.COMBINE.INSTANCES_CONFIDENCE_THRESH
        )

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        # self.relation_head = build_relation_head(cfg)
        # self.panoptic_head = build_panoptic_head(cfg)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):

        # inference
        if not self.training:
            return self.inference(batched_inputs)

        # pre-process
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if "sem_seg" in batched_inputs[0]:
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            gt_sem_seg = None

        # main part
        features = self.backbone(images.tensor)

        # proposal branch
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # roi branch
        gt_mask_logits, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # semantic branch
        sem_seg_logits, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)


        # relation branch

        # panoptic branch

        # loss
        losses = {}
        losses.update(sem_seg_losses)
        losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
        losses.update(proposal_losses)
        return losses

    
    def inference(self, batched_inputs):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, _ = self.proposal_generator(images, features)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        
        detector_results, _ = self.roi_heads(images, features, proposals)
        sem_seg_results, _ = self.sem_seg_head(features)

        processed_results = []
        for sem_seg_result, detector_result, input_per_image, image_size in zip(
            sem_seg_results, detector_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            detector_r = detector_postprocess(detector_result, height, width)

            processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images



def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_limit,
    instances_confidence_threshold,
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0
    segments_info = []

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_confidence_threshold:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": instance_results.pred_classes[inst_id].item(),
                "instance_id": inst_id.item(),
            }
        )

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_limit:
            continue

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": semantic_label,
                "area": mask_area,
            }
        )

    return panoptic_seg, segments_info
