from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.structures import Instances
from detectron2.layers import cat

from .relation_head import build_relation_head
from ..utils import multi_apply, split


def build_panoptic_head(cfg):
    return PanopticHead(cfg)


class PanopticHead(nn.Module):

    def __init__(self, cfg):
        super(PanopticHead, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.ignore_index        = cfg.MODEL.SOGNET.PANOPTIC_HEAD.IGNORE_INDEX
        self.panoptic_los        = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        sem_seg_num_classes      = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.thing_num_classes   = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.stuff_num_classes   = sem_seg_num_classes - self.thing_num_classes
        self.mask_size = 100
        
        # whether add relation process
        self.relation_on = cfg.MODEL.SOGNET.RELATION.ENABLED
        if self.relation_on:
            self.relation_process = build_relation_head(cfg)

    def forward(self, mask_logits, sem_seg_logits, instances, gt_panoptics=None):
        """
        sem_seg_logits: B x C x H x W
        mask_logits: N x C x M x M
        """
        # separate mask_logits
        mask_logits = self._separate_fetch_logits(mask_logits, instances)

        # split sem seg logits
        stuff_logits, thing_logits = split(
            sem_seg_logits, [self.stuff_num_classes, self.thing_num_classes], dim=1)

        if self.training:
            _, pan_losses = multi_apply(
                    self.forward_single,
                    mask_logits,
                    stuff_logits,
                    thing_logits,
                    instances,
                    gt_panoptics)
        else:
            pan_logits, _ = multi_apply(
                    self.forward_single,
                    mask_logits,
                    stuff_logits,
                    thing_logits,
                    instances,
                    gt_panoptics)


    def forward_single(self, mask_logit, stuff_logit, thing_logit, instance, gt_panoptic=None):
        h, w = stuff_logit.size()[-2:]
        thing_mask_logit = self._unmap_mask_logits(mask_logit, instance, (h, w))

        # relation module
        if self.relation_on:
            thing_mask_logit = self.relation_process(thing_mask_logit, instance)

        thing_sem_logit_by_instance = self._crop_thing_logits(thing_logit, instance)
        thing_logit_by_instance = thing_mask_logit + thing_sem_logit_by_instance
        pan_logit = torch.cat([stuff_logit, thing_logit_by_instance], dim=1)

        if not self.training:
            return pan_logit, None
        
        pan_loss = self.pan_loss(pan_logits, gt_panoptic)
        return pan_logits, pan_loss

    def _crop_thing_logits_single(self, thing_sem_seg_logits, bbox, cls_idx):

        h, w = thing_sem_seg_logits.size()[-2:]
        device = thing_sem_seg_logits.device
        num_things = cls_idx.size(0)
        if num_things == 0:
            return torch.ones(1, 1, h, w, device=device) * (-10.)

        thing_logits = torch.zeros(1, num_things, h, w, device=device)
        for i in range(num_things):
            assert cls_idx[i] > 0
            x1 = int(bbox[i, 0])
            y1 = int(bbox[i, 1])
            x2 = int(bbox[i, 2].round() + 1)
            y2 = int(bbox[i, 3].round() + 1)
            thing_logits[0, i, y1: y2, x1: x2] = thing_sem_seg_logits[0, cls_idx[i], y1: y2, x1: x2]

        return thing_logits

    def _unmap_mask_logits_single(self, mask_logits, instance, size):
        bbox = instance.gt_boxes.tensor
        cls_idx = instance.gt_classes

        num_things = cls_idx.size(0)
        thing_mask_logits = torch.zeros((1, num_things) + size, device=self.device)

        if num_things == 0:
            return thing_mask_logits

        bbox = bbox.long()
        bbox_w = bbox[:, 2] - bbox[:, 0] + 1
        bbox_h = bbox[:, 3] - bbox[:, 1] + 1

        for i in range(num_things):
            ref_box = boxes[i, :].long()
            h, w = bbox_h[i], bbox_w[i]
            mask = F.upsample(
                mask_logits[i, 0].view(1, 1, self.mask_size, self.mask_size),
                size=(h, w), mode='bilinear', align_corners=False)
            x0 = max(ref_box[0], 0)
            x1 = min(ref_box[2] + 1, size[1])
            y0 = max(ref_box[1], 0)
            y1 = min(ref_box[3] + 1, size[0])
            thing_mask_logits[0, i, y0: y1, x0: x1] = \
                mask[0, 0, y0 - ref_box[1]: y1 - ref_box[1], x0 - ref_box[0]: x1 - ref_box[0]]
        
        return thing_mask_logits

    def _separate_fetch_logits(self, logits, instances):
        if self.training:
            cls_idx_list = [x.gt_classes for x in instances]
        else:
            cls_idx_list = None

        ins_num_list = [x.size(0) for x in cls_idx_list]
        cls_idx = cat(cls_idx_list)

        logits = logits.gather(1,
               cls_idx.view(-1, 1, 1, 1).expand(-1, -1, self.mask_size, self.mask_size)).squeeze(1)

        return split(logits, ins_num_list)

