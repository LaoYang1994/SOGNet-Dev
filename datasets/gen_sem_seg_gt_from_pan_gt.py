# -*- coding: utf-8 -*-

import time
import functools
import json
import multiprocessing as mp
import numpy as np
import os
from PIL import Image
from functools import reduce

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from fvcore.common.download import download

from panopticapi.utils import rgb2id
from pycocotools.coco import COCO



def _process_panoptic_to_semantic(
    input_panoptic, output_semantic, output_panoptic, segments, img_id, coco_ins, id_map
):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    pan_output = panoptic.copy()

    pan_masks = []
    pan_ids = []
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        mask = panoptic == seg["id"]
        if new_cat_id >= 53:
            pan_masks.append(mask)
            pan_ids.append(seg["id"])
        else:
            pan_output[mask] = new_cat_id
        output[mask] = new_cat_id
    Image.fromarray(output).save(output_semantic)
    Image.fromarray(pan_output).save(output_panoptic)

    ann_ids = coco_ins.getAnnIds(img_id)
    if len(ann_ids) == 0:
        return []

    ret_annos = []
    standard_annos = []
    annos = coco_ins.loadAnns(ann_ids)

    for x in annos:
        if x["iscrowd"] == 1 or x["area"] <= 0:
            x["pan_id"] = -1
            ret_annos.append(x)
        else:
            standard_annos.append(x)

    ins_masks = np.stack([coco_ins.annToMask(x) for x in standard_annos])
    ins_masks = ins_masks.reshape(ins_masks.shape[0], -1)
    pan_masks = np.stack(pan_masks)
    pan_masks = pan_masks.reshape(pan_masks.shape[0], -1)

    inter = (ins_masks[:, None, :] * pan_masks).sum(-1)
    ins_masks_area = ins_masks.sum(-1).reshape(ins_masks.shape[0], 1)
    iou = inter / ins_masks_area

    ins_arg = iou.argmax(axis=1)
    pan_arg = iou.argmax(axis=0)

    ins_mx_pos = ins_arg[:, None] == np.arange(pan_masks.shape[0])
    repeat_pos = ins_mx_pos.sum(axis=0) > 1
    pan_mx_pos = np.arange(ins_masks.shape[0])[:, None] == pan_arg
    pan_mx_pos[:, ~repeat_pos] = True

    selected_pos = (ins_mx_pos & pan_mx_pos).astype(np.float32)

    matched_pan_inds = selected_pos.argmax(axis=1)
    matched_pan_inds[selected_pos.sum(axis=1) == 0] = -1

    for i, x in enumerate(standard_annos):
        if matched_pan_inds[i] == -1:
            x["pan_id"] = -1
        else:
            x["pan_id"] = pan_ids[matched_pan_inds[i]]
        ret_annos.append(x)

    return ret_annos


def separate_coco_semantic_from_panoptic(
    instance_json, panoptic_json, panoptic_root, sem_seg_root, pan_seg_root,
    categories, stuff_only=False):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.

    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.

    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)
    os.makedirs(pan_seg_root, exist_ok=True)

    stuff_ids = [k["id"] for k in categories if k["isthing"] == 0]
    thing_ids = [k["id"] for k in categories if k["isthing"] == 1]
    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(stuff_ids) <= 254
    for i, stuff_id in enumerate(stuff_ids):
        id_map[stuff_id] = i
    for i, thing_id in enumerate(thing_ids):
        if stuff_only:
            id_map[thing_id] = len(stuff_ids)
        else:
            id_map[thing_id] = i + len(stuff_ids)

    with open(panoptic_json) as f:
        obj = json.load(f)

    coco_ins = COCO(instance_json)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for i, anno in enumerate(obj["annotations"]):
            if (i + 1) % 100 == 0:
                print("{} images have been processed!".format(i + 1))
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            sem_output = os.path.join(sem_seg_root, file_name)
            pan_output = os.path.join(pan_seg_root, file_name)

            img_id = anno["image_id"]
            yield input, sem_output, pan_output, segments, img_id

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    new_annos = pool.starmap(
        functools.partial(_process_panoptic_to_semantic, coco_ins=coco_ins, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )

    del coco_ins
    del obj

    new_annos = reduce(lambda x, y: x + y, new_annos)

    new_instances_json = instance_json.replace("train", "sog_train").replace("val", "sog_val")
    print("Start writing to {} ...".format(new_instances_json))

    with open(instance_json) as f:
        coco_anno = json.load(f)
    coco_anno["annotations"] = new_annos

    with open(new_instances_json, "w") as f:
        json.dump(coco_anno, f)

    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "coco")
    # for s in ["val2017", "train2017"]:
    for s in ["val2017"]:
        separate_coco_semantic_from_panoptic(
            os.path.join(dataset_dir, "annotations/instances_{}.json".format(s)),
            os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
            os.path.join(dataset_dir, "panoptic_{}".format(s)),
            os.path.join(dataset_dir, "panoptic_all_cat_{}".format(s)),
            os.path.join(dataset_dir, "panoptic_seg_{}".format(s)),
            COCO_CATEGORIES,
        )
