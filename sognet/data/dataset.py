from detectron2.data import DatasetCatalog, MetadataCatalog


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