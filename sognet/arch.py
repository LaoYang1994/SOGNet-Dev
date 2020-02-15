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

from .modeling import build_panoptic_head

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
        # TODO: build inference
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
        self.panoptic_head = build_panoptic_head(cfg)

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
        # main part
        features = self.backbone(images.tensor)

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

        if "relation" in batched_inputs[0]:
            gt_relations = [x["relation"].to(self.device) for x in batched_inputs]
        else:
            gt_relations = None

        if "pan_seg" in batched_inputs[0]:
            gt_pan_seg = [x["pan_seg"].to(self.device) for x in batched_inputs]
            gt_pan_seg = ImageList.from_tensors(
                gt_pan_seg, self.backbone.size_divisibility, self.panoptic_head.ignore_index
            ).tensor

        else:
            gt_pan_seg = None

        # proposal branch
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            raise NotImplementedError

        # roi branch
        gt_mask_logits, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        # semantic branch
        sem_seg_logits, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)
        # panoptic branch
        _, panoptic_losses = self.panoptic_head(
                gt_mask_logits, sem_seg_logits, gt_instances, gt_relations, gt_pan_seg)

        # loss
        losses = {}
        losses.update(proposal_losses)
        losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
        losses.update(sem_seg_losses)
        # NOTICE: if relation enabled, panoptic_losses contain relation_losses
        losses.update(panoptic_losses)

        return losses

    
    def inference(self, batched_inputs):
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, _ = self.proposal_generator(images, features)
        else:
            raise NotImplementedError
        
        detector_results, _ = self.roi_heads(images, features, proposals)
        sem_seg_results, _ = self.sem_seg_head(features)
        pan_seg_results, _ = self.panoptic_head(None, sem_seg_results, detector_results)

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

