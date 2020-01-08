import torch
import torch.nn as nn


from .utils import multi_apply


def build_panoptic_head(cfg):
    return PanopticHead(cfg)


class PanopticHead(nn.Module):

    def __init__(self, cfg):
        self.ignore_index        = cfg.SOGNET.PANOPTIC_HEAD.IGNORE_INDEX
        self.pan_loss            = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        sem_seg_num_classes      = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.thing_num_classes   = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.stuff_num_classes   = sem_seg_num_classes - self.thing_num_classes

    def forward_single_train(self, sem_seg_logits, mask_logits, things_info, gt_panoptic):
        pass

    def forward_single_test(self, sem_seg_logits, mask_logits):
        pass

    def forward(self, sem_seg_logits, mask_logits, things_info, gt_panoptic=None):
        if self.training:
            assert gt_panoptic is not None
            return multi_apply(self.forward_single_train,
                               sem_seg_logits,
                               mask_logits,
                               things_info,
                               gt_panoptic)
        else:
            return multi_apply(self.forward_single_test,
                               sem_seg_logits,
                               mask_logits,
                               things_info)

    def separate_sem_seg_logits(self, sem_seg_logits, cls_idx, boxes):
        _, h, w    = sem_seg_logits.size()
        device     = sem_seg_logits.device
        num_things = cls_idx.size(0)

        stuff_logits, thing_sem_seg_logits = torch.split(
            sem_seg_logits, [self.stuff_num_classes, self.thing_num_classes], dim=0)

        if num_things == 0:
            thing_logits = torch.ones((0, 0, h, w), dtype=torch.float32, device=device)
        else:
            thing_logits = torch.zeros(
                (num_imgs, num_things, h, w), dtype=torch.float32, device=device)
            for i in range(num_things):
                x1, y1, x2, y2 = boxes[i].round().long()
                x2 += 1
                y2 += 1
                thing_logits[]


