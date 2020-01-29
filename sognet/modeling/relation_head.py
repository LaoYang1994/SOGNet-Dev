import torch
import torch.nn as nn
import torch.nn.functional as F


def build_relation_head(cfg):
    return RelationHead(cfg)


class RelationHead(nn.Module):

    def __init__(self, cfg):
        super(RelationHead, self).__init__()

        self.thing_num_classes = cfg.MODEL.SOGNET.THING_NUM_CLASSES
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

        return torch.cat([dx[..., None], dy[..., None], dw[..., None], dh[..., None]], dim=2)

    def cls_relation(self, cls_idx):
        assert ((cls_idx >= 1) & (cls_idx <= 80)).all()
        one_hot_embedding = F.one_hot(
            cls_idx - 1, self.thing_num_classes - 1).float()

        feat1 = self.U(one_hot_embedding)
        feat2 = self.V(one_hot_embedding)

        cls_relation = feat1[:, None, :] * feat2

        return cls_relation
    
    def position_relation(self, bbox):
        bbox_relative_embedding = self.extract_position_matrix(bbox)
        bbox_relative_embedding = bbox_relative_embedding.permute(
            2, 0, 1).unsqueeze(0)
        bbox_relation = self.W(bbox_relative_embedding)
        bbox_relation = bbox_relation.squeeze(0).permute(1, 2, 0)

        return bbox_relation

    def forward(self, mask_logits, bbox, cls_idx, relation_gt=None):
        device = mask_logits.device
        num_things = cls_idx.size(0)
        if num_things <= 1:
            return mask_logits, torch.tensor([0]).to(mask.device)

        cls_relation = self.cls_relation(cls_idx)
        pos_relation = self.position_relation(bbox)

        relation_feat = torch.cat([cls_relation, pos_relation], dim=2)
        relation_embedding = self.P(relation_feat.permute(2, 0, 1)).unsqueeze(0)
        overlap_score = torch.sigmoid(relation_embedding).squeeze(0).squeeze(0)
        relation_score = F.relu(overlap_score - overlap_score.transpose(0, 1), inplace=True)

        # post process
        mask_prob = torch.sigmoid(mask_logits)
        overlap_part = (mask_logits * mask_prob)[:, :, None, ...] * mask_prob
        overlap_part = overlap_part * relation_score[..., None, None]
        overlap_part = overlap_part.sum(dim=2)
        mask_logits_without_overlap = mask_logits - overlap_part

        if self.training:
            loss_relation = F.mse_loss(overlap_score, relation_gt)
            return mask_logits_without_overlap, loss_relation
        else:
            return mask_logits_without_overlap, None
