import torch
import torch.nn as nn


def build_panoptic_head(cfg):
    return PanopticHead(cfg)


class PanopticHead(nn.Module):

    def __init__(self, cfg):
        pass

    def forward(self, sem_seg_logits, gt_mask_logits, targets):
        pass