# Attention！
This code is based on [detectron2](https://github.com/facebookresearch/detectron2). Unfortunately, There are still some bugs in this code. We will fix these bugs as soon as possible.

# SOGNet
This repository is for [SOGNet: Scene Overlap Graph Network for Panoptic Segmentation](https://arxiv.org/abs/1911.07527) which has been accepted by AAAI2020 and won the Innovation Award in COCO 2019 challenge,

by [Yibo Yang](https://zero-lab-pku.github.io/personwise/yangyibo/), [Hongyang Li](https://zero-lab-pku.github.io/personwise/lihongyang/), [Xia Li](https://zero-lab-pku.github.io/personwise/lixia/), Qijie Zhao, [Jianlong Wu](https://zero-lab-pku.github.io/personwise/wujianlong/), [Zhouchen Lin](https://zero-lab-pku.github.io/personwise/linzhouchen/)

## Introduction
The panoptic segmentation task requires a unified result from semantic and instance segmentation outputs that may contain overlaps. However, current studies widely ignore modeling overlaps. In this study, we aim to model overlap relations among instances and resolve them for panoptic segmentation. Inspired by scene graph representation, we formulate the overlapping problem as a simplified case, named scene overlap graph. We leverage each object's category, geometry and appearance features to perform relational embedding, and output a relation matrix that encodes overlap relations. In order to overcome the lack of supervision, we introduce a differentiable module to resolve the overlap between any pair of instances. The mask logits after removing overlaps are fed into per-pixel instance id classification, which leverages the panoptic supervision to assist in the modeling of overlap relations. Besides, we generate an approximate ground truth of overlap relations as the weak supervision, to quantify the accuracy of overlap relations predicted by our method. Experiments on COCO and Cityscapes demonstrate that our method is able to accurately predict overlap relations, and outperform the state-of-the-art performance for panoptic segmentation. Our method also won the Innovation Award in COCO 2019 challenge.

![SOGNet](assets/sognet.png)

## Usage
+ Pytorch1.4 or above and Python 3 are needed.
+ **We suggest using the [detectron2](https://github.com/LaoYang1994/detectron2) repo that we have forked!!!**.
+ Git the repo to **detectron2/projects** and ```cd SOGNet-Dev```
+ Generate semantic GTs:
```
cd datasets
python gen_coco_sem_seg_gt.py
```
+ Train or Test：
```
sh train.sh or sh test.sh
```

|         |**test split**|**PQ**|**SQ**|**RQ**|**PQ_th**|**PQ_st**|
|---------|--------------|------|------|------|---------|---------|
|SOGNet-50|      val     | 43.7 | 78.7 | 53.5 |  50.6   |   33.1  |

## Citation
If you find SOGNet useful in your research, please consider citing:
```latex
@article{yang19,
 author={Yibo Yang, Hongyang Li, Xia Li, Qijie Zhao, Jianlong Wu, Zhouchen Lin},
 title={SOGNet: Scene Overlap Graph Network for Panoptic Segmentation},
 journaltitle = {{arXiv}:1911.07527 [cs]},
 year={2019}
}
```

## TODO
- [ ] check inference
- [ ] input data: 选择合适的数据组织方式（去掉没有物体的图片）
- [ ] pan head: 貌似少了一个channel
- [ ] 分部分debug，分别用SOG组建出detection, sem_seg, pan_seg三个模块来确定问题
