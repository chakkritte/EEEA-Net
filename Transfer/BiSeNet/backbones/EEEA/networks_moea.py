import torch
import torch.nn as nn
from .operations import *
from torch.autograd import Variable

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, SE=False):
        super(Cell, self).__init__()
        #print(C_prev_prev, C_prev, C)
        self.se_layer = None
        self.nl_layer = None
        self.NL = False

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ActConvBN(C_prev_prev, C, 1, 1)

        self.preprocess1 = ActConvBN(C_prev, C, 1, 1)
    
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)
    
        if SE:
            self.se_layer = SELayer(channel=self.multiplier * C)
        if self.NL:
            self.nl_layer = Nonlocal(self.multiplier * C, nl_c = 1.0, nl_s=1, affine=True)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        if self.se_layer is None:
            return torch.cat([states[i] for i in self._concat], dim=1)
        else:
            #return self.nl_layer(self.se_layer(torch.cat([states[i] for i in self._concat], dim=1)))
            return self.se_layer(torch.cat([states[i] for i in self._concat], dim=1))

class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x

class PyramidNetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, SE=False, increment=4):
        super(PyramidNetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self._increment = increment

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.channels = []
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, SE=SE)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
            C_curr += self._increment

            if i in [self._layers // 3, 2 * self._layers // 3]:
                self.channels.append(C_prev_prev)
        self.channels.append(C_prev)

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        features = []
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i in [self._layers // 3, 2 * self._layers // 3]:
                features.append(s0)     # 8 and 16 times
        features.append(s1)     # 32 times
        return features
