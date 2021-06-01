
# EEEA-Net: An Early Exit Evolutionary Neural Architecture Search

**Submission to Engineering Applications of Artificial Intelligence.**

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


## Usage

### Cloning source code

```
git clone https://github.com/chakkritte/EEEA-Net/
cd EEEA-Net/cifar
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

#### *[name] is mean a name of model

### Architecture evaluation on ImageNet
```
python train_imagenet.py --arch [name]
```

#### *[name] is mean a name of model

## Citation

If you use EEEA-Net or any part of this research, please cite our paper:
```
TBA
```
## License 

Apache-2.0 License
