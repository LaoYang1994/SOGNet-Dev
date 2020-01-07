import os
import json
import time
import argparse
import multiprocessing
import numpy as np

try:
    import pycocotools.coco as COCO
    from pycocotools import mask as COCOmask
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_instance_json_anno')
    parser.add_argument('--coco_panoptic_json_anno')
    args = parser.parse_args()

    return args


def main():
    args = parse()
    ins_anno = COCO(args.coco_instance_json_anno)
    pan_anno = COCO(args.coco_panoptic_json_anno)


if __name__ == "__main__":
    main()