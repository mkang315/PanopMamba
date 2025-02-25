# Official PanopMamba, fine-grained Panoptic Quality (PQ<sup>i</sup>), weighted Panoptic Quality (PQ<sup>w</sup>), frequency-weighted Panoptic Quality (PQ<sup>fw</sup>)
This is the source code for the paper, "PanopMamba: Advancing Nuclei Panoptic Segmentation with Vision State Space Modeling".

## Evaluation
We trained and evaluated PanopMamba on the Hematoxylin and Eosin (H\&E) stained datasets [MoNuSAC2020](https://monusac-2020.grand-challenge.org) and [NuInsSeg](https://doi.org/10.5281/zenodo.10518968).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MoNuSAC2020&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NuInsSeg
| Model | PQ | PQ<sup>i</sup> | PQ<sup>w</sup> | PQ<sup>fw</sup> | PQ | PQ<sup>i</sup> | PQ<sup>w</sup> | PQ<sup>fw</sup> |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | 
| PanopMamba (wo/o C3, replaced by C2f in YOLOv8) | 0.7233 | 0.7320 | 0.7316 | 0.7319 | 0.7508 | 0.7646 | 0.7642 | 0.7646 |
| **PanopMamba (Ours)** | **0.7401** | **0.7515** | **0.7512** | **0.7515** | **0.7838** | **0.8007** | **0.8003** | **0.8007** |
<!--
| [OMG-Seg](https://github.com/lxtGH/OMG-Seg) | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
| [CellViT](https://github.com/TIO-IKIM/CellViT) | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
-->

## License
PanopMamba is released under the MIT License.

## Copyright Notice
Many utility codes of our project base on the codes of [OpenMMLab Semantic Segmentation Toolbox and Benchmark (mmsegmentation)](https://github.com/open-mmlab/mmsegmentation), [Mask2former](https://github.com/facebookresearch/Mask2Former), [MSVMamba](https://github.com/YuHengsss/MSVMamba), [FusionMamba](https://github.com/millieXie/FusionMamba), [MoNuSAC](https://github.com/ruchikaverma-iitg/MoNuSAC/blob/master/PQ_metric.ipynb), [JDTLosses](https://github.com/zifuwanggg/JDTLosses/blob/master/metrics/accuracy_metric.py), [wIoU](https://github.com/engzenia/wIoU), and [FWIoU](https://github.com/15071230989/NFANet/blob/master/FWIoU.py) repositories.
