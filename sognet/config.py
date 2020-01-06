from detectron2.config import CfgNode as CN


def add_sognet_config(cfg):
    """
    Add configs for SOGNet
    """
    _C = cfg
    _C.MODEL.SOGNET = CN()
