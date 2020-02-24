# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .transforms import annotations_to_instances, get_relation_gt, pan_id2channel_id, get_fcn_roi_gt

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["SOGDatasetMapper"]


class SOGDatasetMapper:

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.relation_on    = cfg.MODEL.SOGNET.RELATION.ENABLED
        self.fcn_roi_on     = cfg.MODEL.SOGNET.FCN_ROI.ENABLED
        # fmt: on

        self.fcn_roi_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2

        self.stuff_cls_num  = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES - cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        ).contiguous()
        # Can use uint8 if it turns out to be slow some day

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pan_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            instances = utils.filter_empty_instances(instances)
            # TODO: design a data structure for storing relation mat
            if self.relation_on:
                instances, gt_relation = get_relation_gt(instances)
                dataset_dict["relation"] = gt_relation

            if instances.has("bit_masks"):
                instances.remove("bit_masks")

            dataset_dict["instances"] = instances

        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            if self.fcn_roi_on:
                fcn_roi_gt = get_fcn_roi_gt(sem_seg_gt, dataset_dict["instances"], self.fcn_roi_on)
                dataset_dict["fcn_roi"] = fcn_roi_gt
            dataset_dict["sem_seg"] = sem_seg_gt

        if "pan_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("pan_seg_file_name"), "rb") as f:
                pan_seg_gt = Image.open(f)
                pan_seg_gt = np.asarray(pan_seg_gt, dtype="uint32")

            pan_seg_gt = pan_id2channel_id(
                    pan_seg_gt, dataset_dict["instances"].pan_id, self.stuff_cls_num)
            dataset_dict["instances"].remove("pan_id")
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
            pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))
            dataset_dict["pan_seg"] = pan_seg_gt

        return dataset_dict

