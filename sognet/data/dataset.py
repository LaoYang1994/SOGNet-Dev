import os
import copy

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES


def merge_to_panoptic_sog(detection_dicts, sem_seg_dicts, pan_dicts):
    results = []
    sem_seg_file_to_entry = {x["file_name"]: x for x in sem_seg_dicts}
    pan_file_to_entry = {x["file_name"]: x for x in pan_dicts}
    assert len(sem_seg_file_to_entry) > 0

    for det_dict in detection_dicts:
        dic = copy.copy(det_dict)
        dic.update(sem_seg_file_to_entry[dic["file_name"]])
        dic.update(pan_file_to_entry[dic["file_name"]])
        results.append(dic)
    return results

def get_sog_metadata():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    stuff_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    assert len(stuff_ids) == 53, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 53], used in models) to ids in the dataset (used for processing results)
    # The id 0 is mapped to an extra category "thing".
    stuff_dataset_id_to_contiguous_id = {k: i + 1 for i, k in enumerate(stuff_ids)}
    # When converting COCO panoptic annotations to semantic annotations
    # We label the "thing" category to 0
    stuff_dataset_id_to_contiguous_id[0] = 0

    # 54 names for COCO stuff categories (including "things")
    stuff_classes = ["things"] + [
        k["name"].replace("-other", "").replace("-merged", "")
        for k in COCO_CATEGORIES
        if k["isthing"] == 0
    ]

    # NOTE: I randomly picked a color for things
    stuff_colors = [[82, 18, 128]] + [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    ret.update(thing_ret)
    return ret


_PREDEFINED_SPLITS_COCO_PANOPTIC_SOG = {
    "coco_2017_train_panoptic_sog": (
        "coco/train2017",
        # generate by running the scripts in datasets
        "coco/annotations/instances_sog_train2017.json",
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        "coco/panoptic_all_cat_train2017",
        ["pan_id"],
    ),
    "coco_2017_val_panoptic_sog": (
        "coco/val2017",
        "coco/annotations/instances_sog_val2017.json",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_all_cat_val2017",
        [],
    )
}


for (
    prefix, (image_root, instances_json, panoptic_root, panoptic_json, semantic_root, extra_keys),
) in _PREDEFINED_SPLITS_COCO_PANOPTIC_SOG.items():

    sog_metadata   = get_sog_metadata()
    image_root     = os.path.join("datasets", image_root)
    instances_json = os.path.join("datasets", instances_json)
    panoptic_root  = os.path.join("datasets", panoptic_root)
    panoptic_json  = os.path.join("datasets", panoptic_json)
    sem_seg_root   = os.path.join("datasets", semantic_root)

    DatasetCatalog.register(
        prefix,
        lambda: merge_to_panoptic_sog(
            load_coco_json(instances_json, image_root, prefix, extra_keys),
            load_sem_seg(sem_seg_root, image_root),
            {}
        ),
    )
    MetadataCatalog.get(prefix).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        sem_seg_root=sem_seg_root,
        json_file=instances_json,  # TODO rename
        evaluator_type="coco_panoptic_seg",
        **sog_metadata
    )
