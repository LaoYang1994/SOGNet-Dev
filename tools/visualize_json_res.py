import os
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sem_json_res')
    parser.add_argument('--output_dir')

    global args
    args = parser.parse_args()


def main():
    parse()

    color_map = {}
    for x in COCO_CATEGORIES:
        color_map[x['id']] = x['color']

    with open(args.sem_json_res) as f:
        segs = json.load(f)

    file_name_seg_map = defaultdict(list)
    for seg in segs:
        file_name_seg_map[seg['file_name']].append(seg)

    for path, segs in file_name_seg_map.items():
        file_name = os.path.basename(path).replace('png', 'jpg')
        target_path = os.path.join(args.output_dir, 'sem_seg', file_name)
        h, w = segs[0]['segmentation']['size']
        png = np.zeros((h, w, 3), dtype=np.uint8)

        for x in segs:
            cat_id = x['category_id']
            color = color_map[cat_id]
            mask = maskUtils.decode(x['segmentation'])
            for i in range(3):
                png[..., i][mask] = color[i]
        
        Image.fromarray(png).save(target_path)



