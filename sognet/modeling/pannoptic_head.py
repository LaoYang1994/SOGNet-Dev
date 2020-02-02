from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.structures import Instances
from .relation_head import build_relation_head

def build_panoptic_head(cfg):
    return PanopticHead(cfg)


class PanopticHead(nn.Module):

    def __init__(self, cfg):
        super(PanopticHead, self).__init__()
        self.ignore_index        = cfg.MODEL.SOGNET.PANOPTIC_HEAD.IGNORE_INDEX
        self.pan_loss            = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        sem_seg_num_classes      = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.thing_num_classes   = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.stuff_num_classes   = sem_seg_num_classes - self.thing_num_classes
        self.mask_size = 100
        self.relation_process = build_relation_head(cfg)

    def forward_single(self,
                       stuff_logits,
                       thing_sem_seg_logits,
                       mask_logits,
                       boxes,
                       cls_idx,
                       gt_panoptic=None):
        _, _, h, w = thing_sem_seg_logits.size()
        thing_mask_logits = self._unmap_mask_logits(mask_logits, boxes, cls_idx, (h, w))
        thing_sem_seg_logits_by_instance = self._crop_thing_logits(thing_sem_seg_logits,
                                                                   boxes, cls_idx)
        thing_logits_by_instance = thing_mask_logits + thing_sem_seg_logits_by_instance
        pan_logits = torch.cat([stuff_logits, thing_logits_by_instance], dim=1)

        if not self.training:
            return pan_logits, None
        
        pan_loss = self.pan_loss(pan_logits, gt_panoptic)
        return pan_logits, pan_loss

    def forward(self,
                sem_seg_logits:torch.Tensor,
                mask_logits:List[torch.Tensor],
                gt_instances=None ,
                gt_panoptic=None):
        """
        sem_seg_logits: N x C x H x W
        mask_logits: N x C x M x M
        """
        # return
        pan_logits = []
        pan_losses = []
        # parse information
        num_imgs = sem_seg_logits.size(0)
        gt_boxes_list = [x.gt_boxes for x in gt_instances]
        gt_classes_list = [x.gt_classes for x in gt_instances]

        # split sem seg logits
        stuff_logits, thing_sem_seg_logits_by_class = torch.split(
            sem_seg_logits, [self.stuff_num_classes, self.thing_num_classes], dim=1)

        for i in range(num_imgs):
            # prepare input
            gt_boxes_i     = gt_boxes_list[i]
            gt_classes_i   = gt_classes_list[i]
            mask_logits_i  = mask_logits[i]
            stuff_logits_i = stuff_logits[[i]]
            thing_sem_seg_logits_by_class_i = thing_sem_seg_logits_by_class[[i]]

            pan_logit_single, pan_loss_single = self.forward_single(stuff_logits_i,
                                                                    thing_sem_seg_logits_by_class_i,
                                                                    mask_logits_i,
                                                                    gt_boxes_i,
                                                                    gt_classes_i)
            pan_logits.append(pan_logit_single)
            pan_losses.append(pan_loss_single)

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

    def _unmap_mask_logits_single(self, mask_logits, bbox, cls_idx, size):
        device = mask_logits.device
        num_things = cls_idx.size(0)
        thing_mask_logits = torch.zeros((1, num_things) + size, device=device)

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
