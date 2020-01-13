import torch
import torch.nn as nn

def build_panoptic_head(cfg):
    return PanopticHead(cfg)


class PanopticHead(nn.Module):

    def __init__(self, cfg):
        self.ignore_index        = cfg.SOGNET.PANOPTIC_HEAD.IGNORE_INDEX
        self.pan_loss            = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        sem_seg_num_classes      = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.thing_num_classes   = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.stuff_num_classes   = sem_seg_num_classes - self.thing_num_classes

    def forward_single(self,
                       stuff_logits,
                       thing_sem_seg_logits,
                       mask_logits,
                       boxes,
                       cls_idx,
                       gt_panoptic=None):
        _, _, h, w = sem_seg_logits.size()
        thing_mask_logits = self._unmap_mask_logits(mask_logits, boxes, cls_idx, (h, w))
        thing_sem_seg_logits_by_instance = self._crop_thing_logits(thing_sem_seg_logits,
                                                                   boxes, cls_idx)
        thing_logits_by_instance = thing_mask_logits + thing_sem_seg_logits_by_instance
        pan_logits = torch.cat([stuff_logits, thing_logits_by_instance], dim=1)

        if not self.training:
            return pan_logits, None
        
        pan_loss = self.pan_loss(pan_logits, gt_panoptic)
        return pan_logits, pan_loss

    def forward(self, sem_seg_logits, mask_logits, gt_instances=None, gt_panoptic=None):
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

        stuff_logits, thing_sem_seg_logits_by_class = torch.split(
            sem_seg_logits, [self.stuff_num_classes, self.thing_num_classes], dim=1)

        for i in range(num_imgs):
            gt_boxes = gt_boxes_list[i]
            gt_classes = gt_classes_list[i]
            stuff_logits_i = stuff_logits[[i]]
            thing_sem_seg_logits_by_class_i = thing_sem_seg_logits_by_class[[i]]
            mask_logits_i = mask_logits[i]
            pan_logit_single, pan_loss_single = self.forward_single(stuff_logits_i,
                                                                    thing_sem_seg_logits_by_class_i,
                                                                    mask_logits_i,
                                                                    gt_boxes,
                                                                    gt_classes)
            pan_logits.append(pan_logit_single)
            pan_losses.append(pan_loss_single)

    def _crop_thing_logits(self, thing_sem_seg_logits_by_class, bbox, cls_idx):
        pass

    def _unmap_mask_logits(self, mask_logits, bbox, cls_idx, size):
        pass