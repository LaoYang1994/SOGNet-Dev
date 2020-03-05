from typing import Dict
import fvcore.nn.weight_init as weight_init

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers.deform_conv import DeformConv
from detectron2.layers.roi_align import ROIAlign
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.modeling.meta_arch.semantic_seg import SEM_SEG_HEADS_REGISTRY


class DeformConvWithOffset(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False):
        super(DeformConvWithOffset, self).__init__()
        
        self.conv_offset = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size * 2 * deformable_groups,
            kernel_size=3,
            stride=1,
            padding=1)
        self.deform_conv = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x):
        offset = self.conv_offset(x)
        x = self.deform_conv(x, offset)

        return x


class DeformConvFCNSubNet(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        deformable_group=1,
        dilation=1,
        with_norm='none'):
        super(DeformConvFCNSubNet, self).__init__()

        assert with_norm in ['none', 'GN']
        assert num_layers >= 2
        self.num_layers = num_layers
        if with_norm == 'GN':
            def group_norm(in_channel):
                return nn.GroupNorm(32, in_channel)
            norm = group_norm
        else:
            norm = None
        self.conv = nn.ModuleList()
        for i in range(num_layers):
            conv = []
            if i == num_layers - 2:
                conv.append(DeformConvWithOffset(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation))
                in_channels = out_channels
            else:
                conv.append(DeformConvWithOffset(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation))
            if with_norm != 'none':
                conv.append(norm(in_channels))
            conv.append(nn.ReLU(inplace=True))
            self.conv.append(nn.Sequential(*conv))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, DeformConv):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv[i](x)
        return x


@SEM_SEG_HEADS_REGISTRY.register()
class XDCNSemSegFPNHead(nn.Module):

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):

        super(XDCNSemSegFPNHead, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features         = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value        = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes              = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims                = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        num_layers               = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        self.common_stride       = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm                     = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.sem_seg_loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT

        self.fcn_roi_on       = cfg.MODEL.SOGNET.FCN_ROI.ENABLED
        if self.fcn_roi_on:
            self.fcn_roi_loss_weight   = cfg.MODEL.SOGNET.FCN_ROI.LOSS_WEIGHT
            pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2
            pooler_scales     = (1.0 / self.common_stride, )
            sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
            pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE

            self.roi_pooler = ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type
            )

        self.fcn_subnet = DeformConvFCNSubNet(
            feature_channels[self.in_features[0]], conv_dims, num_layers, with_norm=norm)

        self.predictor = nn.Conv2d(
            len(self.in_features) * conv_dims, num_classes,kernel_size=1, stride=1, padding=0)
        self._weight_init()

    def _weight_init(self):
        nn.init.normal_(self.predictor.weight.data, 0, 0.01)
        self.predictor.bias.data.zero_()

    def forward(self, features, gt_sem_seg=None, instances=None, gt_fcn_roi=None):
        seg_feature_list = []
        for i, f_name in enumerate(self.in_features):
            seg_feature = self.fcn_subnet(features[f_name])
            if i > 0:
                seg_feature = F.interpolate(
                    seg_feature, scale_factor=2**i, mode="bilinear", align_corners=False)
            seg_feature_list.append(seg_feature)
        seg_features = torch.cat(seg_feature_list, dim=1)

        sem_seg_scores = self.predictor(seg_features)
        sem_seg_scores_init_size = F.interpolate(
                sem_seg_scores, scale_factor=self.common_stride,
                mode="bilinear", align_corners=False)

        if self.training:
            assert gt_sem_seg is not None
            losses = {}
            losses["loss_sem_seg"] = (
                F.cross_entropy(
                    sem_seg_scores_init_size, gt_sem_seg,
                    reduction="mean", ignore_index=self.ignore_value)
                * self.sem_seg_loss_weight
            )
            if self.fcn_roi_on:
                assert gt_fcn_roi is not None
                assert instances is not None
                if gt_fcn_roi.size(0) == 0:
                    dummy_size = gt_fcn_roi.size()[-2:]
                    dummy_size = (len(instances), ) + dummy_size
                    gt_fcn_roi = torch.zeros(dummy_size).type_as(gt_fcn_roi) + 255
                    dummy_box = torch.tensor([[0., 0., 1., 1.]]).to(self.device)
                    rois = [Boxes(dummy_box) for i in range(len(instances))] 
                else:
                    rois = [x.gt_boxes for x in instances]
                roi_feats = self.roi_pooler([seg_features], rois)
                roi_scores = self.predictor(roi_feats)
                fcn_roi_loss = F.cross_entropy(
                    roi_scores, gt_fcn_roi, reduction="mean",
                    ignore_index=self.ignore_value
                )
                losses.update({"loss_fcn_roi": fcn_roi_loss * self.fcn_roi_loss_weight})
            return sem_seg_scores, losses
        else:
            return sem_seg_scores_init_size, {}
