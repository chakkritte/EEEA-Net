
# EEEA-Net: An Early Exit Evolutionary Neural Architecture Search

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eeea-net-an-early-exit-evolutionary-neural/neural-architecture-search-on-cifar-10)](https://paperswithcode.com/sota/neural-architecture-search-on-cifar-10?p=eeea-net-an-early-exit-evolutionary-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eeea-net-an-early-exit-evolutionary-neural/object-detection-on-pascal-voc-2007)](https://paperswithcode.com/sota/object-detection-on-pascal-voc-2007?p=eeea-net-an-early-exit-evolutionary-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eeea-net-an-early-exit-evolutionary-neural/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=eeea-net-an-early-exit-evolutionary-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eeea-net-an-early-exit-evolutionary-neural/neural-architecture-search-on-imagenet)](https://paperswithcode.com/sota/neural-architecture-search-on-imagenet?p=eeea-net-an-early-exit-evolutionary-neural)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eeea-net-an-early-exit-evolutionary-neural/image-classification-on-cifar-100)](https://paperswithcode.com/sota/image-classification-on-cifar-100?p=eeea-net-an-early-exit-evolutionary-neural)

**This paper has been published to Engineering Applications of Artificial Intelligence.**

Paper: [EAAI version](https://www.sciencedirect.com/science/article/pii/S0952197621002451) or [arXiv version](https://arxiv.org/pdf/2108.06156.pdf)

This implementation of EEEA-Net (Early Exit Evolutionary Algorithm Network) from EEEA-Net: An Early Exit Evolutionary Neural Architecture Search by [Chakkrit Termritthikun](https://chakkritte.github.io/cv/), et al.

<p align="center">
  <img src="img/early_exit.JPG" alt="early exit">
</p>


**This code is based on the implementation of  [DARTS](https://github.com/quark0/darts), [NSGA-Net](https://github.com/ianwhale/nsga-net), [NSGA-Net-v2](https://github.com/mikelzc1990/nsganetv2), [Once for All](https://github.com/mit-han-lab/once-for-all), and [TransferLearning-Tasks](https://github.com/EMI-Group/TransferLearning-Tasks).**

## Results

<p align="center">
  <img src="img/result_imagenet.JPG" alt="imagenet">
</p>

## Prerequisite for server
 - Tested on Ubuntu OS version 18.04.x
 - Tested on PyTorch 1.6 and TorchVision 0.7.0


## Quick Usage (EEEA-Net, ImageNet pre-trained)

#### install darmo package
```
pip install darmo
```

#### import darmo and create model; see more models at [darmo](https://github.com/jitdee-ai/darmo)
```
import darmo
model = darmo.create_model("eeea_c2", num_classes=1000, pretrained=True)
```

#### supported transfer learning
```
model.reset_classifier(num_classes=100, dropout=0.2)
```

## Usage

### Cloning source code

```
git clone https://github.com/chakkritte/EEEA-Net/
cd EEEA-Net/EEEA/cifar
```

### Install Requirements

```
pip install -r requirements.txt
```

### Architecture search on CIFAR-10 (Normal search)

```
python search_space.py --dataset cifar10 --search normal --th_param 0.0 
```

### Architecture search on CIFAR-10 (Early Exit search with beta equal 5)

```
python search_space.py --dataset cifar10 --search ee --th_param 5.0 
```

### Architecture evaluation on CIFAR-10 

```
python train_cifar.py --arch [name]
```

#### *[name] is mean a name of models [EA, EEEA_A, EEEA_B, EEEA_C]


## Citation

If you use EEEA-Net or any part of this research, please cite our paper:
```
  @article{termritthikun2021EEEANet,
  title = "{EEEA-Net: An Early Exit Evolutionary Neural Architecture Search}",
  journal = {Engineering Applications of Artificial Intelligence},
  volume = {104},
  pages = {104397},
  year = {2021},
  author = {Chakkrit Termritthikun and Yeshi Jamtsho and Jirarat Ieamsaard and Paisarn Muneesawang and Ivan Lee},
  }
```
## License 

Apache-2.0 License
