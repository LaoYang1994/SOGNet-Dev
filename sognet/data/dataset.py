import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata


def get_sog_metadata():
    pass

_PREDEFINED_SPLITS_COCO_PANOPTIC_SOG = {
    "coco_2017_train_panoptic_sog": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_all_cat_train2017",
    ),
    "coco_2017_val_panoptic_sog": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_all_cat_val2017",
    )
}


for (
    prefix,
    (panoptic_root, panoptic_json, semantic_root),
) in _PREDEFINED_SPLITS_COCO_PANOPTIC_SOG.items():
    prefix_instances = prefix[: -len("_panoptic_sog")]
    instances_meta = MetadataCatalog.get(prefix_instances)
    image_root, instances_json = instances_meta.image_root, instances_meta.json_file
    register_coco_panoptic_separated(
        prefix,
        _get_builtin_metadata("coco_panoptic_separated"),
        image_root,
        os.path.join("datasets", panoptic_root),
        os.path.join("datasets", panoptic_json),
        os.path.join("datasets", semantic_root),
        instances_json,
    )
