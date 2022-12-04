# VICRegL: Self-Supervised Learning of Local Visual Features

This repository provides a PyTorch implementation and pretrained models for VICRegL, a self-supervsied pretraining method for learning global and local features, described in the paper [VICRegL: Self-Supervised Learning of Local Visual Features](https://arxiv.org/abs/2210.01571), published to NeurIPS 2022.\
Adrien Bardes, Jean Ponce and Yann LeCun\
Meta AI, Inria

--- 

<p align="center">
<img src=".github/vicregl_archi.jpg" width=100% height=100% 
class="center">
</p>

## Pre-trained Models

You can choose to download only the weights of the pretrained backbone used for downstream tasks, or the full checkpoint which contains backbone and expander/projector weights. All the models are pretrained on ImageNet-1k, except the ConvNeXt-XL model which is pretrained on ImageNet-22k. **linear cls.** is the linear classification accuracy on the validation set of ImageNet, and **linear seg.** is the linear frozen mIoU on the validation set of Pascal VOC.

<table>
  <tr>
    <th>arch</th>
    <th>alpha</th>
    <th>params</th>
    <th>linear cls. (%)</th>
    <th>linear seg. (mIoU)</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>0.9</td>
    <td>23M</td>
    <td>71.2</td>
    <td>54.0</td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.9.pth">backbone</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.9_fullckpt.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.9_stats.txt">logs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>0.75</td>
    <td>23M</td>
    <td>70.4</td>
    <td>55.9</td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.75.pth">backbone</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.75_fullckpt.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.75_stats.txt">logs</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-S</td>
    <td>0.9</td>
    <td>50M</td>
    <td>75.9</td>
    <td>66.7</td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.9.pth">backbone</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.9_fullckpt.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.9_stats.txt">logs</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-S</td>
    <td>0.75</td>
    <td>50M</td>
    <td>74.6</td>
    <td>67.5</td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.75.pth">backbone</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.75_fullckpt.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.75_stats.txt">logs</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-B</td>
    <td>0.9</td>
    <td>85M</td>
    <td>77.1</td>
    <td>69.3</td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_base_alpha0.9.pth">backbone</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_base_alpha0.9_fullckpt.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_base_alpha0.9_stats.txt">logs</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-B</td>
    <td>0.75</td>
    <td>85M</td>
    <td>76.3</td>
    <td>70.4</td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_base_alpha0.75.pth">backbone</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_base_alpha0.75_fullckpt.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_base_alpha0.75_stats.txt">logs</a></td>
  </tr>
  <tr>
    <td>ConvNeXt-XL</td>
    <td>0.75</td>
    <td>350M</td>
    <td>79.4</td>
    <td>78.7</td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_xlarge_alpha0.75.pth">backbone</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_xlarge_alpha0.75_fullckpt.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/vicregl/convnext_xlarge_alpha0.75_stats.txt">logs</a></td>
  </tr>
  
</table>

## Pretrained models on PyTorch Hub

```python
import torch
model = torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p9')
model = torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p75')
model = torch.hub.load('facebookresearch/vicregl:main', 'convnext_small_alpha0p9')
model = torch.hub.load('facebookresearch/vicregl:main', 'convnext_small_alpha0p75')
model = torch.hub.load('facebookresearch/vicregl:main', 'convnext_base_alpha0p9')
model = torch.hub.load('facebookresearch/vicregl:main', 'convnext_base_alpha0p75')
model = torch.hub.load('facebookresearch/vicregl:main', 'convnext_xlarge_alpha0p75')
```


## Training

Install PyTorch ([pytorch.org](http://pytorch.org)) and download [ImageNet](https://imagenet.stanford.edu/). The code has been developed for PyTorch version 1.8.1 and torchvision version 0.9.1, but should work with other versions just as well. Setup the ImageNet path in the file [datasets.py](https://github.com/facebookresearch/VICRegL/datasets.py):

```
IMAGENET_PATH = "path/to/imagenet"
```

ImageNet can also be loaded from numpy files, by setting the flag `--dataset_from_numpy` and setting the path:

```
IMAGENET_NUMPY_PATH = "path/to/imagenet/numpy/files"
```

The argument `--alpha` controls the weight between global and local loss, it is set by default to 0.75.

### Single-node local training

To pretrain VICRegL with a ResNet-50 backbone on a single node with 8 GPUs for 100 epochs, run:

```
python -m torch.distributed.launch --nproc_per_node=8 main_vicregl.py --fp16 --exp-dir /path/to/experiment/ --arch resnet50 --epochs 100 --batch-size 512 --optimizer lars --base-lr 0.3 --weight-decay 1e-06 --size-crops 224 --num-crops 2 --min_scale_crops 0.08 --max_scale_crops 1.0 --alpha 0.75
```

To pretrain VICRegL with a ConvNeXt-S backbone, run:

```
python -m torch.distributed.launch --nproc_per_node=8 main_vicregl.py --fp16 --exp-dir /path/to/experiment/ --arch convnext_small --epochs 100 --batch-size 384 --optimizer adamw --base-lr 0.00075 --alpha 0.75
```

### Multi-node training with SLURM

To pretrain VICRegL with a ResNet-50 backbone, with [submitit](https://github.com/facebookincubator/submitit) (`pip install submitit`) and SLURM on 4 nodes with 8 GPUs each for 300 epochs, run:

```
python run_with_submitit.py --nodes 4 --ngpus 8 --fp16 --exp-dir /path/to/experiment/ --arch resnet50 --epochs 300 --batch-size 2048 --optimizer lars --base-lr 0.2 --weight-decay 1e-06 --size-crops 224 --num-crops 2 --min_scale_crops 0.08 --max_scale_crops 1.0 --alpha 0.75
```

To pretrain VICRegL with a ConvNeXt-B backbone, run:

```
python run_with_submitit.py --nodes 2 --ngpus 8 --fp16 --exp-dir /path/to/experiment/ --arch convnext_small --epochs 400 --batch-size 576 --optimizer adamw --base-lr 0.0005 --alpha 0.75
```


## Evaluation

### Linear evaluation

To evaluate a pretrained backbone (resnet50, convnext_small, convnext_base, convnext_xlarge) on linear classification on ImageNet, run:

```
python evaluate.py --data-dir /path/to/imagenet/ --pretrained /path/to/checkpoint/model.pth --exp-dir /path/to/experiment/ --arch [backbone] --lr-head [lr]
```

with `lr=0.02` for resnets models and `lr=0.3` for convnexts models.

### Linear segmentation

See the segmentation folder.

## License

This project is released under the CC-BY-NC License. See [LICENSE](LICENSE) for details.

## Citation
If you find this repository useful, please consider giving a star :star: and citation:

```
@inproceedings{bardes2022vicregl,
  author  = {Adrien Bardes and Jean Ponce and Yann LeCun},
  title   = {VICRegL: Self-Supervised Learning of Local Visual Features},
  booktitle = {NeurIPS},
  year    = {2022},
}
```
