import os
import torch
import logging
import detectron2.utils.comm as comm
from collections import OrderedDict
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch
)
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results
)

from sognet import (
    add_sognet_config,
    SOGDatasetMapper,
    COCOPanopticEvaluatorWith2ChPNG
)

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        # semantic seg evaluation
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder
            )
        )
        # detection evaluation
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        # panoptic evaluation
        evaluator_list.append(
            COCOPanopticEvaluatorWith2ChPNG(
                dataset_name, output_folder, gen_png=cfg.MODEL.SOGNET.GEN_PNG))

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=SOGDatasetMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=SOGDatasetMapper(cfg, False))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sognet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
