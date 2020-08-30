# -*- coding: utf-8 -*-
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
        self.stuff_area_limit = cfg.MODEL.SOGNET.POSTPROCESS.STUFF_AREA_LIMIT
        self.stuff_num_classes = (cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES - 
                                  cfg.MODEL.ROI_HEADS.NUM_CLASSES)

        self.combine_on = cfg.MODEL.SOGNET.COMBINE.ENABLED
        if self.combine_on:
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
        pixel_std  = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
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

        # prepare gt
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

        if "fcn_roi" in batched_inputs[0]:
            gt_fcn_roi = [x["fcn_roi"].to(self.device) for x in batched_inputs]
            gt_fcn_roi = torch.cat(gt_fcn_roi, dim=0)
        else:
            gt_fcn_roi = None

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
        sem_seg_logits, sem_roi_losses = self.sem_seg_head(
            features, gt_sem_seg, gt_instances, gt_fcn_roi)
        # panoptic branch
        _, pan_relation_losses = self.panoptic_head(
                gt_mask_logits, sem_seg_logits, gt_instances, gt_relations, gt_pan_seg)

        # loss
        losses = {}
        losses.update(proposal_losses)
        losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
        # NOTICE: if fcn_roi enabled, sem_roi_losses contain fcn_roi_losses
        losses.update(sem_roi_losses)
        # NOTICE: if relation enabled, pan_relation_losses contain relation_losses
        losses.update(pan_relation_losses)

        return losses

    def inference(self, batched_inputs):
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, _ = self.proposal_generator(images, features)
        else:
            raise NotImplementedError
        
        detector_results, pan_detector_results = self.roi_heads(images, features, proposals)
        sem_seg_results, _ = self.sem_seg_head(features)
        pan_seg_results, _ = self.panoptic_head(None, sem_seg_results, pan_detector_results)

        processed_results = []
        for sem_seg_result, detector_result, pan_seg_result, input_per_image, image_size in zip(
            sem_seg_results, detector_results, pan_seg_results, batched_inputs, images.image_sizes
        ):
            processed_result = {}
            height     = input_per_image.get("height")
            width      = input_per_image.get("width")
            sem_seg_r  = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            detector_r = detector_postprocess(detector_result, height, width)
            processed_result.update({"sem_seg": sem_seg_r,
                                      "instances": detector_r})

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold,
                )
            else:
                pan_pred = sem_seg_postprocess(
                        pan_seg_result["pan_logit"], image_size, height, width)
                del pan_seg_result["pan_logit"]
                pan_seg_result["pan_pred"] = pan_pred.argmax(dim=0)
                panoptic_r = pan_seg_postprocess(
                    pan_seg_result,
                    sem_seg_r.argmax(dim=0),
                    self.stuff_num_classes,
                    self.stuff_area_limit)
            processed_result.update({"panoptic_seg": panoptic_r})

            processed_results.append(processed_result)
        return processed_results

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


def pan_seg_postprocess(
        panoptic_results, semantic_results, stuff_num_classes=53, stuff_area_limit=64**2):

    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)
    segments_info = []
    current_segment_id = 0

    pan_results = panoptic_results["pan_pred"]
    pred_classes = panoptic_results["pan_ins_cls"]

    area_ids = pan_results.unique()
    inst_ids = area_ids[area_ids >= stuff_num_classes]

    for inst_id in inst_ids:
        mask = pan_results == inst_id
        sem_cls, area = semantic_results[mask].unique(return_counts=True)
        sem_pred_cls = sem_cls[area.argmax()]
        pan_pred_cls = pred_classes[inst_id - stuff_num_classes]
        if sem_pred_cls == pan_pred_cls + stuff_num_classes:
            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": True,
                    "category_id": pan_pred_cls.item(),
                    "instance_id": inst_id.item() - stuff_num_classes,
                }
            )
        else:
            if area.max() / area.sum() >= 0.5 and sem_pred_cls < stuff_num_classes:
                pan_results[mask] = sem_pred_cls
            else:
                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id
                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": True,
                        "category_id": pan_pred_cls.item(),
                        "instance_id": inst_id.item() - stuff_num_classes,
                    }
                )
    area_ids = pan_results.unique()
    sem_ids = area_ids[area_ids < stuff_num_classes]
    for sem_id in sem_ids:
        mask = (pan_results == sem_id) & (panoptic_seg == 0)
        if mask.sum() < stuff_area_limit:
            continue
        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": sem_id.item(),
                "area": mask.sum().item(),
            }
        )

    return panoptic_seg, segments_info


def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_limit,
    instances_confidence_threshold,
):
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
