from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import (ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads,
        select_foreground_proposals, build_box_head, build_mask_head)
from detectron2.structures import Boxes, ImageList, Instances

# from .rcnn_customized import FastRCNNOutputLayers, mask_rcnn_inference


@ROI_HEADS_REGISTRY.register()
class SOGROIHeads(StandardROIHeads):

    def __init__(self, cfg, input_shape):
        super(SOGROIHeads, self).__init__(cfg, input_shape)
        # self.sog_test_score_thresh = cfg.MODEL.SOGNET.POSTPROCESS.INSTANCES_CONFIDENCE_THRESH

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            gt_mask_logits = self.forward_for_logits_with_given_boxes(features, targets)

            del targets
            return gt_mask_logits, losses
        else:
            det_instances, pan_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            det_instances = self.forward_with_given_boxes(features, det_instances)
            pan_instances = self.forward_with_given_boxes(features, pan_instances)
            return det_instances, pan_instances

    def forward_for_logits_with_given_boxes(self, features, instances):
        """
        Get the feature of the given boxes in `instances`.
        """
        features = [features[f] for f in self.in_features]

        if self.training:
            boxes = [x.gt_boxes for x in instances]
        else:
            boxes = [x.pred_boxes for x in instances]
        mask_features = self.mask_pooler(features, boxes)
        mask_logits = self.mask_head.layers(mask_features)

        return mask_logits
