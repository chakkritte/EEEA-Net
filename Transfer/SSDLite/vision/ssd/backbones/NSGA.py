from timm.models.layers import drop_path
from ofa.imagenet_codebase.modules.layers import *
from ofa.layers import set_layer_from_config, MBInvertedConvLayer, ConvLayer, IdentityLayer, LinearLayer
from ofa.imagenet_codebase.utils import MyNetwork, make_divisible
from ofa.imagenet_codebase.networks.proxyless_nets import MobileInvertedResidualBlock
import torch

class MobileInvertedResidualBlock(MyModule):
    """
    Modified from https://github.com/mit-han-lab/once-for-all/blob/master/ofa/
    imagenet_codebase/networks/proxyless_nets.py to include drop path in training

    """
    def __init__(self, mobile_inverted_conv, shortcut, drop_connect_rate=0.0):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            # res = self.mobile_inverted_conv(x) + self.shortcut(x)
            res = self.mobile_inverted_conv(x)

            if self.drop_connect_rate > 0.:
                res = drop_path(res, drop_prob=self.drop_connect_rate, training=self.training)

            res += self.shortcut(x)

        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(
            mobile_inverted_conv, shortcut, drop_connect_rate=config['drop_connect_rate'])

class MyNetwork(MyModule):

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def zero_last_gamma(self):
        raise NotImplementedError
    
    """ implemented methods """
    
    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

class MobileNetV3(MyNetwork):

    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier, depth):
        super(MobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.depth = depth
        self.channels = []

    def forward(self, x):
        feats = []
        n_blocks = [sum(self.depth[:2]) , sum(self.depth[:4])]
        x = self.first_conv(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            if i in n_blocks:
                feats.append(x)
                self.channels.append(x.size(1))

        x = self.final_expand_layer(x)

        feats.append(x)
        self.channels.append(x.size(1))

        #x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        #x = self.feature_mix_layer(x)
        #x = torch.squeeze(x)
        #x = self.classifier(x)
        return feats
        # if self.classifier is not None:
        #     x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        #     x = self.classifier(x)
        #     return x
        # else:
        #     return feats

    
class NSGANetV2(MobileNetV3):
    """
    Modified from https://github.com/mit-han-lab/once-for-all/blob/master/ofa/
    imagenet_codebase/networks/mobilenet_v3.py to include drop path in training
    and option to reset classification layer
    """
    @staticmethod
    def build_from_config(config, depth=[], drop_connect_rate=0.0):
        first_conv = set_layer_from_config(config['first_conv'])
        final_expand_layer = set_layer_from_config(config['final_expand_layer'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_idx, block_config in enumerate(config['blocks']):
            block_config['drop_connect_rate'] = drop_connect_rate * block_idx / len(config['blocks'])
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))
        
    
        net = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier, depth)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

    @staticmethod
    def reset_classifier(model, last_channel, n_classes, dropout_rate=0.0):
        model.classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

def eeeanet(weight_path=None):
    config = json.load(open(weight_path+'/net.config'))
    subnet = json.load(open(weight_path+'/net.subnet'))
    model = NSGANetV2.build_from_config(config, depth=subnet['d'])
    if weight_path is not None:
        init = torch.load(weight_path+'/net.inherited', map_location='cpu')['state_dict']
        model.load_state_dict(init)
    return model