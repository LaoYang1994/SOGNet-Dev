import os
import copy
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from fvcore.common.file_io import PathManager

logger = logging.getLogger(__name__)


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


def load_pan_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["pan_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


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
    stuff_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 0] + thing_ids
    assert len(stuff_ids) == 133, len(stuff_ids)

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    # When converting COCO panoptic annotations to semantic annotations
    # We label the "thing" category to 0

    # 54 names for COCO stuff categories (including "things")
    stuff_classes = [
        k["name"].replace("-other", "").replace("-merged", "")
        for k in COCO_CATEGORIES
        if k["isthing"] == 0
    ] + thing_classes

    # NOTE: I randomly picked a color for things
    stuff_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 0] + (thing_colors)
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
        "coco/panoptic_seg_train2017",
        "coco/annotations/panoptic_train2017.json",
        "coco/panoptic_all_cat_train2017",
        ["pan_id"],
    ),
    "coco_2017_val_panoptic_sog": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_all_cat_val2017",
        [],
    ),
    "coco_2017_val_panoptic_sog_debug": (
        "coco/val2017",
        "coco/annotations/instances_sog_val2017.json",
        "coco/panoptic_seg_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_all_cat_val2017",
        ["pan_id"],
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
            load_pan_seg(panoptic_root, image_root)
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
