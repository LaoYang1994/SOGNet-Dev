import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import cat


def build_relation_head(cfg):
    return RelationHead(cfg)


class RelationHead(nn.Module):

    def __init__(self, cfg):
        super(RelationHead, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.thing_num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.cls_embedding_dim = cfg.MODEL.SOGNET.RELATION.CLS_EMBEDDING_DIM
        self.pos_embedding_dim = cfg.MODEL.SOGNET.RELATION.POS_EMBEDDING_DIM
        self.relation_embedding_dim = self.cls_embedding_dim + self.pos_embedding_dim

        self.U = nn.Sequential(
                nn.Linear(self.thing_num_classes, self.cls_embedding_dim, bias=False),
                nn.ReLU(inplace=True))
        self.V = nn.Sequential(
                nn.Linear(self.thing_num_classes, self.cls_embedding_dim, bias=False),
                nn.ReLU(inplace=True))
        self.W = nn.Sequential(
                nn.Conv2d(4, self.pos_embedding_dim, 1, bias=False),
                nn.ReLU(inplace=True))
        self.P = nn.Conv2d(self.relation_embedding_dim, 1, 1, bias=False)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.P.weight)

    def extract_position_matrix(self, rois):
        x_min = rois[:, 0]
        y_min = rois[:, 1]
        x_max = rois[:, 2]
        y_max = rois[:, 3]
        w = x_max - x_min
        h = y_max - y_min
        x_ctr = (x_min + x_max) / 2.
        y_ctr = (y_min + y_max) / 2.

        dx = -(x_ctr[:, None] - x_ctr)
        dx = dx / w[:, None]

        dy = -(y_ctr[:, None] - y_ctr)
        dy = dy / h[:, None]

        dw = torch.log(w / w[:, None])
        dh = torch.log(h / h[:, None])

        return torch.stack([dx, dy, dw, dh])

    def cls_relation(self, cls_idx):
        one_hot_embedding = F.one_hot(
            cls_idx, self.thing_num_classes).float()

        feat1 = self.U(one_hot_embedding)
        feat2 = self.V(one_hot_embedding)

        cls_relation = feat1[:, None, :] * feat2

        return cls_relation.permute(2, 0, 1).unsqueeze(0)
    
    def position_relation(self, bbox):
        bbox_relative_embedding = self.extract_position_matrix(bbox)
        bbox_relative_embedding = bbox_relative_embedding.unsqueeze(0)
        bbox_relation = self.W(bbox_relative_embedding)
        bbox_relation = bbox_relation

        return bbox_relation

    def forward(self, mask_logit, instance, gt_relation=None):

        if not self.training:
            return self.inference(mask_logit, instance)

        assert gt_relation is not None
        relation_num = gt_relation.size(0)

        assert len(instance) == 0 or relation_num <= len(instance)

        if relation_num == 1:
            cls_idx = torch.arange(1).to(self.device).type_as(instance.gt_classes)
            bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]]).to(self.device)
            relation_score = self.relation_predict(cls_idx, bbox)
            mask_logit_wo_overlap = mask_logit
        else:
            bbox = instance.gt_boxes.tensor[:relation_num]
            cls_idx = instance.gt_classes[:relation_num]
            relation_score = self.relation_predict(cls_idx, bbox)

            mask_logit, sep_mask_logit = torch.split(
                    mask_logit, [relation_num, len(instance)-relation_num], dim=1)
            mask_logit_wo_overlap = self.duplicate_removal(mask_logit, relation_score)
            mask_logit_wo_overlap = torch.cat([mask_logit_wo_overlap, sep_mask_logit], dim=1)

        overlap_score = relation_score + relation_score.transpose(0, 1)
        loss_relation = F.mse_loss(overlap_score, gt_relation)

        return mask_logit_wo_overlap, loss_relation

    def inference(self, mask_logit, instance):
        bbox = instance.pred_boxes.tensor
        cls_idx = instance.pred_classes

        relation_score = self.relation_predict(cls_idx, bbox)
        mask_logit_wo_overlap = self.duplicate_removal_loop(mask_logit, relation_score)

        return mask_logit_wo_overlap, {}

    def relation_predict(self, cls_idx, bbox):

        cls_relation = self.cls_relation(cls_idx)
        pos_relation = self.position_relation(bbox)

        relation_feat = torch.cat([cls_relation, pos_relation], dim=1)
        relation_embedding = self.P(relation_feat)
        overlap_score = torch.sigmoid(relation_embedding).squeeze(0).squeeze(0)
        relation_score = F.relu(overlap_score - overlap_score.transpose(0, 1), inplace=True)

        return relation_score

    def duplicate_removal(self, mask_logit, relation_score):
        mask_prob = torch.sigmoid(mask_logit)
        overlap_part = (mask_logit * mask_prob)[:, :, None, ...] * mask_prob
        overlap_part = overlap_part * relation_score[..., None, None]
        overlap_part = overlap_part.sum(dim=2)
        mask_logit_wo_overlap = mask_logit - overlap_part

        return mask_logit_wo_overlap

    def duplicate_removal_loop(self, mask_logit, relation_score):
        mask_area = torch.sigmoid(mask_logit) * mask_logit
        mask_logit_wo_overlap = mask_logit

        num_things = mask_logit.size(1)

        for i in range(num_things):
            overlap_part = mask_area[0, [i]] * mask_logit[0]
            overlap_part = overlap_part * relation_score[i].reshape(-1, 1, 1).sum(dim=0)
            mask_logit_wo_overlap[0, i] -= overlap_part

        return mask_logit_wo_overlap

