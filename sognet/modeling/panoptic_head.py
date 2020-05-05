from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from detectron2.structures import Instances
from detectron2.layers import cat
from detectron2.layers.mask_ops import _do_paste_mask

from .relation_head import build_relation_head
from ..utils import multi_apply, reduce_loss

BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024 ** 3


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
            # thing_mask_logit = self._unmap_mask_logit_single(mask_logit, instance, feat_size)
            thing_mask_logit = self.paste_mask_logit_in_image(
                    mask_logit, instance.gt_boxes.tensor / self.feat_stride, feat_size)
        else:
            thing_mask_logit, instance = self._unmap_mask_removal(mask_logit, instance, feat_size)

        # relation module
        if self.relation_on:
            if self.training or len(instance):
                thing_mask_logit, relation_loss = self.relation_process(
                        thing_mask_logit, instance, gt_relation)
        else:
            relation_loss = {}

        thing_sem_logit = self._crop_thing_logit_single(thing_logit, instance)
        thing_logit = thing_mask_logit + thing_sem_logit
        pan_logit = torch.cat([stuff_logit[None, ...], thing_logit], dim=1)

        if not self.training:
            pan_result = {"pan_logit": pan_logit,
                          "pan_ins_cls": instance.pred_classes}
            return pan_result, {}, {}
        
        gt_panoptic = F.interpolate(
                gt_panoptic[None, None, ...].float(), size=feat_size).squeeze(1).long()
        pan_loss = self.panoptic_loss(pan_logit, gt_panoptic)

        return pan_logit, relation_loss, pan_loss

    def paste_mask_logit_in_image(self, mask_logits, boxes, image_shape):
        assert mask_logits.shape[-1] == mask_logits.shape[-2], \
            "Only square mask predictions are supported"
        N = mask_logits.size(0)
        if N == 0:
            return mask_logits.new_empty((0,) + image_shape, dtype=torch.float)
        if not isinstance(boxes, torch.Tensor):
            boxes = boxes.tensor
        assert len(boxes) == N, boxes.shape

        img_h, img_w = image_shape

        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if self.device.type == "cpu":
            # CPU is most efficient when they are pasted one by one with skip_empty=True
            # so that it performs minimal number of operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks, but may have memory issue
            num_chunks = int(np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (
                num_chunks <= N
            ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
        chunks = torch.chunk(torch.arange(N, device=self.device), num_chunks)

        img_mask_logits = torch.zeros(
            N, img_h, img_w, device=self.device, dtype=torch.float
        )
        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_logits[inds, None, :, :], boxes[inds], img_h, img_w,
                skip_empty=self.device.type == "cpu"
            )

            img_mask_logits[(inds,) + spatial_inds] = masks_chunk
        return img_mask_logits

    def _unmap_mask_removal(self, mask_logit, instance, size):
        num_things = len(instance)
        # TODO: check whether it's needed!
        if num_things == 0:
            thing_mask_logit = torch.zeros((1, 1) + size, device=self.device)
            return thing_mask_logit, instance

        boxes   = instance.pred_boxes
        # Maybe there is a bug in this place.
        pred_classes = instance.pred_classes

        thing_mask_logit = self.paste_mask_logit_in_image(mask_logit, boxes, size)
        mask_panel = torch.zeros((self.thing_num_classes, ) + size, 
                dtype=torch.bool, device=self.device)

        score = instance.scores
        sorted_inds = score.argsort(descending=True)

        left_flag = torch.ones(num_things)

        for inst_id in sorted_inds:
            cls_idx = pred_classes[inst_id]
            logit = thing_mask_logit[inst_id]
            bit_mask = logit > 0

            mask_area = bit_mask.sum()
            cls_panel = mask_panel[cls_idx]

            if (mask_area == 0) or (
                (cls_panel & bit_mask).sum().float() / 
                    mask_area.float() > self.removal_thresh):
                left_flag[inst_id] = -1
                continue

            mask_panel[cls_idx] |= bit_mask

        left_inds = (left_flag > 0).nonzero().flatten()

        instance = instance[left_inds]
        thing_mask_logit = thing_mask_logit[left_inds]

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

        if num_things == 0:
            thing_logit = torch.ones(1, 1, h, w, device=self.device) * (-10)
            return thing_logit

        thing_logit = torch.zeros(num_things, h, w, device=self.device)

        if self.training:
            bbox = bbox / self.feat_stride
        # TODO: This is different upsnet, to check!
        bbox = bbox.round().int()

        for i in range(num_things):
            # TODO: check whether cls_idx > 0
            x1, y1, x2, y2 = bbox[i]
            thing_logit[i, y1: y2, x1: x2] = thing_sem_logit[cls_idx[i], y1: y2, x1: x2]

        return thing_logit[None, ...]

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
