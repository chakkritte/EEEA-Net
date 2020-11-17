import os
from .DARTS.networks import NetworkImageNet
from .DARTS import genotypes

from .EEEA.networks_moea import PyramidNetworkImageNet
from .EEEA import genotypes_moea

from .NASNet.network import NASNetAMobile
from .Mnasnet import MNASNet
from .MobileNetV2 import MobileNetV2
from .MobileNetV3 import MobileNetV3
from .ShuffleNetV2 import ShuffleNetV2
from .ResNet import ResNet, BasicBlock, Bottleneck
from .NSGA import NSGANetV2
import json

def nsgac1(weight_path=None):
    print(weight_path)
    config = json.load(open(weight_path+'/net.config'))
    subnet = json.load(open(weight_path+'/net.subnet'))
    model = NSGANetV2.build_from_config(config, depth=subnet['d'])
    return model

def nsgac2(weight_path=None):
    print(weight_path)
    config = json.load(open(weight_path+'/net.config'))
    subnet = json.load(open(weight_path+'/net.subnet'))
    model = NSGANetV2.build_from_config(config, depth=subnet['d'])

    return model

def eeeal(weight_path=None):
    genotype = eval("genotypes_moea.%s" % 'EEEAL')
    base_net = PyramidNetworkImageNet(48, 1000, 14, False, genotype, SE=True, increment=8)
    return base_net

def moea(weight_path=None):
    genotype = eval("genotypes_moea.%s" % 'MOEA')
    base_net = PyramidNetworkImageNet(48, 1000, 14, False, genotype, SE=True, increment=8)
    return base_net

def moea16(weight_path=None):
    genotype = eval("genotypes_moea.%s" % 'MOEA')
    base_net = PyramidNetworkImageNet(48, 1000, 14, False, genotype, SE=True, increment=16)
    return base_net

def moea_sota(weight_path=None):
    genotype = eval("genotypes_moea.%s" % 'MOEA_SOTA_PI')
    base_net = PyramidNetworkImageNet(48, 1000, 14, False, genotype, SE=True, increment=8)
    return base_net

def moea_sota12(weight_path=None):
    genotype = eval("genotypes_moea.%s" % 'MOEA_SOTA_PI')
    base_net = PyramidNetworkImageNet(48, 1000, 14, False, genotype, SE=True, increment=12)
    return base_net

def pairnas(weight_path=None):
    genotype = eval("genotypes.%s" % 'PairNAS_CIFAR10')
    base_net = NetworkImageNet(46, 1000, 14, False, genotype)
    return base_net


def darts(weight_path=None):
    genotype = eval("genotypes.%s" % 'DARTS')
    base_net = NetworkImageNet(48, 1000, 14, False, genotype)
    return base_net


def nasnet(weight_path=None):
    base_net = NASNetAMobile()
    return base_net


def mnasnet(weight_path=None, **kwargs):    # 1.0
    model = MNASNet(1.0, **kwargs)
    return model


def mobilenetv2(weight_path=None):
    model = MobileNetV2()
    return model

def mobilenetv3(weight_path=None):
    model = MobileNetV3()
    return model

def shufflenetv2(weight_path=None):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024])   # 1.0
    return model

def resnet18(weight_path=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(weight_path=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(weight_path=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model