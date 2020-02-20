from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.structures import Instances
from detectron2.layers import cat

from .relation_head import build_relation_head
from ..utils import multi_apply, reduce_loss


def build_panoptic_head(cfg):
    return PanopticHead(cfg)


class PanopticHead(nn.Module):

    def __init__(self, cfg):
        super(PanopticHead, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.ignore_index        = cfg.MODEL.SOGNET.PANOPTIC.IGNORE_INDEX
        sem_seg_num_classes      = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.thing_num_classes   = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.stuff_num_classes   = sem_seg_num_classes - self.thing_num_classes
        # TODO: maybe add a global config is better
        self.mask_size           = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2
        self.feat_stride         = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        self.removal_thresh      = cfg.MODEL.SOGNET.PANOPTIC.REMOVAL_THRESH
        
        # panoptic loss
        self.panoptic_loss       = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.pan_loss_weight     = cfg.MODEL.SOGNET.PANOPTIC.LOSS_WEIGHT

        # whether add relation process
        self.relation_on = cfg.MODEL.SOGNET.RELATION.ENABLED
        if self.relation_on:
            self.relation_process = build_relation_head(cfg)
            self.relation_loss_weight = cfg.MODEL.SOGNET.RELATION.LOSS_WEIGHT

    def forward(self, mask_logits, sem_seg_logits, instances, gt_relations=None, gt_panoptics=None):
        # split sem seg logits
        stuff_logits, thing_logits = torch.split(
            sem_seg_logits, [self.stuff_num_classes, self.thing_num_classes], dim=1)

        if self.training:
            # separate mask_logits, special for training. During inference, the masks have been
            # separated.
            mask_logits = self._separate_fetch_logits(mask_logits, instances)

            losses = {}
            if self.relation_on:
                _, relation_losses, pan_losses = multi_apply(self.forward_single,
                    mask_logits, stuff_logits, thing_logits, instances, gt_panoptics, gt_relations)

                relation_losses = reduce_loss(relation_losses, reduction="mean")
                losses.update({"loss_relation": relation_losses * self.relation_loss_weight})
            else:
                _, _, pan_losses = multi_apply(self.forward_single,
                    mask_logits, stuff_logits, thing_logits, instances, gt_panoptics)

            pan_losses = reduce_loss(pan_losses, reduction="mean")
            losses.update({"loss_panoptic": pan_losses * self.pan_loss_weight})
            return None, losses
        else:
            mask_logits = []
            for instance in instances:
                mask_logits.append(instance.mask_logit)
                instance.remove("mask_logit")
            pan_results, _, _ = multi_apply(self.forward_single,
                    mask_logits, stuff_logits, thing_logits, instances)
            return pan_results, {}

    def forward_single(
        self, mask_logit, stuff_logit, thing_logit, instance, gt_panoptic=None, gt_relation=None):

        feat_size = stuff_logit.size()[-2:]

        if self.training:
            thing_mask_logit = self._unmap_mask_logit_single(mask_logit, instance, feat_size)
        else:
            thing_mask_logit, instance = self._unmap_mask_removal(mask_logit, instance, feat_size)

        # relation module
        if self.relation_on and len(instance) > 0:
            thing_mask_logit, relation_loss = self.relation_process(
                    thing_mask_logit, instance, gt_relation)
        else:
            relation_loss = {}

        thing_sem_logit = self._crop_thing_logit_single(thing_logit, instance)
        thing_logit = thing_mask_logit + thing_sem_logit
        pan_logit = torch.cat([stuff_logit[None, ...], thing_logit], dim=1)

        if not self.training:
            pan_result = {"pan_pred": pan_logit.argmax(dim=1)[0],
                          "pan_ins_cls": instance.pred_classes}
            return pan_result, {}, {}
        
        gt_panoptic = F.interpolate(
                gt_panoptic[None, None, ...].float(), size=feat_size).squeeze(1).long()
        pan_loss = self.panoptic_loss(pan_logit, gt_panoptic)

        return pan_logit, relation_loss, pan_loss

    def _unmap_mask_logit_single(self, mask_logit, instance, size):
        bbox = instance.gt_boxes.tensor
        cls_idx = instance.gt_classes

        num_things = cls_idx.size(0)
        thing_mask_logit = torch.zeros((1, num_things) + size, device=self.device)

        if num_things == 0:
            return thing_mask_logit

        bbox = bbox / self.feat_stride
        bbox = bbox.long()
        bbox_w = bbox[:, 2] - bbox[:, 0] + 1
        bbox_h = bbox[:, 3] - bbox[:, 1] + 1

        # TODO: In this place, roi upsample maybe is better
        for i in range(num_things):
            ref_box = bbox[i]
            h, w = bbox_h[i], bbox_w[i]
            mask = F.interpolate(
                mask_logit[i].view(1, 1, self.mask_size, self.mask_size),
                size=(h, w), mode='bilinear', align_corners=False)
            x0 = max(ref_box[0], 0)
            x1 = min(ref_box[2] + 1, size[1])
            y0 = max(ref_box[1], 0)
            y1 = min(ref_box[3] + 1, size[0])
            thing_mask_logit[0, i, y0: y1, x0: x1] = (
                    mask[0, 0, y0 - ref_box[1]: y1 - ref_box[1], x0 - ref_box[0]: x1 - ref_box[0]])
        
        return thing_mask_logit

    def _unmap_mask_removal(self, mask_logit, instance, size):
        bbox = instance.pred_boxes.tensor
        cls_idx = instance.pred_classes
        score = instance.scores
        # for mask removal
        mask_panel = torch.zeros((self.thing_num_classes, ) + size, dtype=torch.bool, device=self.device)

        num_things = len(instance)
        thing_mask_logit = torch.zeros((1, num_things) + size, device=self.device)

        if num_things == 0:
            return thing_mask_logit, instance

        order = (-score).argsort()
        bbox = bbox[order]
        cls_idx = cls_idx[order]
        mask_logit = mask_logit[order]

        bbox = bbox.long()
        bbox_wh = bbox[:, 2:] - bbox[:, :2] + 1

        # TODO: In this place, roi upsample maybe is better
        for i in range(num_things):
            ref_box = bbox[i]
            h, w = bbox_wh[i]
            logit = F.interpolate(
                mask_logit[i].view(1, 1, self.mask_size, self.mask_size),
                size=(h, w), mode='bilinear', align_corners=False)[0, 0]
            mask = logit > 0

            x0 = max(ref_box[0], 0)
            x1 = min(ref_box[2] + 1, size[1])
            y0 = max(ref_box[1], 0)
            y1 = min(ref_box[3] + 1, size[0])

            crop_mask = mask[y0 - ref_box[1]: y1 - ref_box[1], x0 - ref_box[0]: x1 - ref_box[0]]
            mask_area = crop_mask.sum()
            crop_mask_panel = mask_panel[cls_ids[i], y0: y1, x0: x1]

            if (mask_area == 0) or (
                (crop_mask_panel & mask).sum().float() / mask_area.float() > self.removal_thresh):
                order[i] = -1
                continue

            mask_panel[cls_idx[i], y0: y1, x0: x1] |= crop_mask
            thing_mask_logit[0, i, y0: y1, x0: x1] = (
                    mask[0, 0, y0 - ref_box[1]: y1 - ref_box[1], x0 - ref_box[0]: x1 - ref_box[0]])

        order_inds = (order > 0).nonzero().reshape(-1)
        instance = instance[order_inds]
        
        return thing_mask_logit, instance

    def _crop_thing_logit_single(self, thing_sem_logit, instance):

        if self.training:
            bbox = instance.gt_boxes.tensor
            cls_idx = instance.gt_classes
        else:
            bbox = instance.pred_boxes.tensor
            cls_idx = instance.pred_classes

        h, w = thing_sem_logit.size()[-2:]
        num_things = cls_idx.size(0)

        thing_logit = torch.zeros(1, num_things, h, w, device=self.device)
        if num_things == 0:
            return thing_logit

        bbox = bbox / self.feat_stride

        for i in range(num_things):
            # TODO: check whether cls_idx > 0
            x1 = int(bbox[i, 0])
            y1 = int(bbox[i, 1])
            x2 = int(bbox[i, 2].round() + 1)
            y2 = int(bbox[i, 3].round() + 1)
            thing_logit[0, i, y1: y2, x1: x2] = thing_sem_logit[cls_idx[i], y1: y2, x1: x2]

        return thing_logit

    def _separate_fetch_logits(self, logits, instances):
        if self.training:
            cls_idx_list = [x.gt_classes for x in instances]
        else:
            cls_idx_list = None

        ins_num_list = [x.size(0) for x in cls_idx_list]
        cls_idx = cat(cls_idx_list)

        logits = logits.gather(1,
               cls_idx.view(-1, 1, 1, 1).expand(-1, -1, self.mask_size, self.mask_size)).squeeze(1)

        return torch.split(logits, ins_num_list)

