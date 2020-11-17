import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import math

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'conv_1x1_3x3' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C//2, (1,1), stride=(1, 1), padding=(0, 0), bias=False),
    nn.BatchNorm2d(C//2, affine=affine),
    nn.ReLU(inplace=False),
    nn.Conv2d(C//2, C, (3,3), stride=(stride, stride), padding=(1, 1), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  "conv_3x1_1x3" : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,3), stride=(1, stride), padding=(0, 1), bias=False),
    nn.Conv2d(C, C, (3,1), stride=(stride, 1), padding=(1, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),

  'inv_res_3x3' : lambda C, stride, affine: InvertedResidual(C, C, 3, stride, expand_ratio=1, affine=affine, shuffle=False),
  'inv_res_5x5' : lambda C, stride, affine: InvertedResidual(C, C, 5, stride, expand_ratio=1, affine=affine, shuffle=False),
  
  'inv_res_3x3_sh' : lambda C, stride, affine: InvertedResidual(C, C, 3, stride, expand_ratio=1, affine=affine, shuffle=True),
  'inv_res_5x5_sh' : lambda C, stride, affine: InvertedResidual(C, C, 5, stride, expand_ratio=1, affine=affine, shuffle=True),

  #'std_gn_3x3' : lambda C, stride, affine: ReLUStdConvGN(C, C, 3, stride, 1, affine=affine),
  'std_gn_3x3' : lambda C, stride, affine: ActConvBN(C, C, 3, stride, affine=affine, preact=True, conv_layer=StdConv2d, norm_layer=nn.GroupNorm, act_layer=nn.ReLU),

  #'std_gn_5x5' : lambda C, stride, affine: ReLUStdConvGN(C, C, 5, stride, 2, affine=affine),
  'std_gn_5x5' : lambda C, stride, affine: ActConvBN(C, C, 5, stride, affine=affine, preact=True, conv_layer=StdConv2d, norm_layer=nn.GroupNorm, act_layer=nn.ReLU),

  # 'std_gn_7x7' : lambda C, stride, affine: ReLUStdConvGN(C, C, 7, stride, 3, affine=affine),
  'std_gn_7x7' : lambda C, stride, affine: ActConvBN(C, C, 7, stride, affine=affine, preact=True, conv_layer=StdConv2d, norm_layer=nn.GroupNorm, act_layer=nn.ReLU),

  'mbconv_k3_t1': lambda C, stride, affine: MBConv(C, C, 3, stride, 1, t=1, affine=affine),
  'mbconv_k5_t1': lambda C, stride, affine: MBConv(C, C, 5, stride, 2, t=1, affine=affine),
  'mbconv_k7_t1': lambda C, stride, affine: MBConv(C, C, 7, stride, 3, t=1, affine=affine),

  'octave_conv_3x3': lambda C, stride, affine: ReLU_OctaveConv_BN(C, C, kernel_size=3, stride=stride, padding=1, affine=affine, alpha_in=0, alpha_out=0),
  'octave_conv_5x5': lambda C, stride, affine: ReLU_OctaveConv_BN(C, C, kernel_size=5, stride=stride, padding=2, affine=affine, alpha_in=0, alpha_out=0),
  'blur_pool_3x3' : lambda C, stride, affine: BlurPool2d(C, filt_size=3, stride=stride),
}

class ActConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, dilation=1, groups=1, affine=True, preact=True, conv_layer=nn.Conv2d, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
    super(ActConvBN, self).__init__()
    
    self.preact = preact
    padding = self._get_padding(kernel_size, stride, dilation)  # assuming PyTorch style padding for this block
    
    if self.preact:
        if act_layer is not None:
            self.act = act_layer(inplace=False)
        else:
            self.act = None
            
    self.conv = conv_layer(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)

    if norm_layer == nn.GroupNorm:
        self.norm = norm_layer(C_out//2, C_out, affine=affine)
    else:
        self.norm = norm_layer(C_out, affine=affine)
        
    if not self.preact:
        if act_layer is not None:
            self.act = act_layer(inplace=True)
        else:
            self.act = None
    
  # Calculate symmetric padding for a convolution
  def _get_padding(self, kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2
        
  def forward(self, x):
    if self.preact and self.act is not None:
        x = self.act(x)
    x = self.conv(x)
    x = self.norm(x)
    if not self.preact and self.act is not None:
        x = self.act(x)
    return x

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class InvertedResidual(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, expand_ratio ,affine=True, shuffle=False, preact=True, act_layer=nn.ReLU):
    super(InvertedResidual, self).__init__()
    self.expand_ratio = expand_ratio
    self.stride = stride
    self.use_res_connect = self.stride == 1 and C_in == C_out
    self.shuffle = shuffle
    self.op = nn.Sequential(
      # pw
      ActConvBN(C_in, C_in * self.expand_ratio, 1, 1, affine=affine, preact=preact, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act_layer=act_layer),
      # dw
      ActConvBN(C_in * self.expand_ratio, C_in * self.expand_ratio, kernel_size, stride, groups=C_in * self.expand_ratio, affine=affine, preact=preact, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act_layer=act_layer),
      # pw-linear
      ActConvBN(C_in * self.expand_ratio, C_out, 1, 1, affine=True, preact=preact, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act_layer=None),
      )

  def forward(self, x):
    if self.use_res_connect:
      if self.shuffle:
        out = x + self.op(x)
        out = channel_shuffle(out, 2)
      else:
        out = x + self.op(x)
      return out
    else:
      if self.shuffle:
        out = self.op(x)
        out = channel_shuffle(out, 2)
      else:
        out = self.op(x)
      return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class StdConv2d(nn.Conv2d):

  def forward(self, x):
    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    w = (w - m) / torch.sqrt(v + 1e-10)
    return F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)

#https://github.com/JaminFong/DenseNAS
class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, t=3, affine=True, use_se=False, preact=True, act_layer=nn.ReLU):
        super(MBConv, self).__init__()
        self.t = t
        if self.t > 1:
            self._expand_conv = ActConvBN(C_in, C_in*self.t, 1, 1, groups=1, affine=affine, preact=preact, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act_layer=act_layer)

            self._depthwise_conv = ActConvBN(C_in*self.t, C_in*self.t, kernel_size, stride, groups=C_in*self.t, affine=affine, preact=preact, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act_layer=act_layer)

            self._project_conv = ActConvBN(C_in*self.t, C_out, 1, 1, groups=1, affine=affine, preact=preact, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act_layer=None)
        else:
            self._expand_conv = None

            self._depthwise_conv = ActConvBN(C_in, C_in, kernel_size, stride, groups=C_in, affine=affine, preact=preact, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act_layer=act_layer)

            self._project_conv = ActConvBN(C_in, C_out, 1, 1, groups=1, affine=affine, preact=preact, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act_layer=None)

    def forward(self, x):
        input_data = x
        if self._expand_conv is not None:
            x = self._expand_conv(x)
        x = self._depthwise_conv(x)
        out_data = self._project_conv(x)

        if out_data.shape == input_data.shape:
            return out_data + input_data
        else:
            return out_data

#https://github.com/d-li14/octconv.pytorch
class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 else \
                        nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        if x_h is not None:
            x_h = self.downsample(x_h) if self.stride == 2 else x_h
            x_h2h = self.conv_h2h(x_h)
            x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 else None
        if x_l is not None:
            x_l2h = self.conv_l2h(x_l)
            x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None 
            x_h = x_l2h + x_h2h
            x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
            return x_h, x_l
        else:
            return x_h2h, x_h2l

class ReLU_OctaveConv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, affine=True):
        super(ReLU_OctaveConv_BN, self).__init__()
        self.act = activation_layer(inplace=False)
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)), affine=affine)
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out), affine=affine)
    
    def forward(self, x):
        x = self.act(x)
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h

# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class BlurPool2d(nn.Module):
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling
    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride
    Returns:
        torch.Tensor: the transformed tensor.
    """
    filt: Dict[str, torch.Tensor]

    def __init__(self, channels, filt_size=3, stride=2) -> None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        pad_size = [get_padding(filt_size, stride, dilation=1)] * 4
        self.padding = nn.ReflectionPad2d(pad_size)
        self._coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs)  # for torchscript compat
        self.filt = {}  # lazy init by device for DataParallel compat

    def _create_filter(self, like: torch.Tensor):
        blur_filter = (self._coeffs[:, None] * self._coeffs[None, :]).to(dtype=like.dtype, device=like.device)
        return blur_filter[None, None, :, :].repeat(self.channels, 1, 1, 1)

    def _apply(self, fn):
        # override nn.Module _apply, reset filter cache if used
        self.filt = {}
        super(BlurPool2d, self)._apply(fn)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        C = input_tensor.shape[1]
        blur_filt = self.filt.get(str(input_tensor.device), self._create_filter(input_tensor))
        return F.conv2d(
            self.padding(input_tensor), blur_filt, stride=self.stride, groups=C)


class ZeroInitBN(nn.BatchNorm2d):
    """BatchNorm with zero initialization."""

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

class Nonlocal(nn.Module):
    """Lightweight Non-Local Module.
    See https://arxiv.org/abs/2004.01961
    """

    def __init__(self, n_feature, nl_c, nl_s, affine=None):
        super(Nonlocal, self).__init__()
        self.n_feature = n_feature
        self.nl_c = nl_c
        self.nl_s = nl_s
        self.depthwise_conv = nn.Conv2d(n_feature,
                                        n_feature,
                                        3,
                                        1, (3 - 1) // 2,
                                        groups=n_feature,
                                        bias=False)

        self.bn = ZeroInitBN(n_feature, affine=affine)

    def forward(self, l):
        N, n_in, H, W = list(l.shape)
        reduced_HW = (H // self.nl_s) * (W // self.nl_s)
        l_reduced = l[:, :, ::self.nl_s, ::self.nl_s]
        theta, phi, g = l[:, :int(self.nl_c * n_in), :, :], l_reduced[:, :int(
            self.nl_c * n_in), :, :], l_reduced
        if (H * W) * reduced_HW * n_in * (1 + self.nl_c) < (
                H * W) * n_in**2 * self.nl_c + reduced_HW * n_in**2 * self.nl_c:
            f = torch.einsum('niab,nicd->nabcd', theta, phi)
            f = torch.einsum('nabcd,nicd->niab', f, g)
        else:
            f = torch.einsum('nihw,njhw->nij', phi, g)
            f = torch.einsum('nij,nihw->njhw', f, theta)
        f = f / H * W
        f = self.bn(self.depthwise_conv(f))
        return f + l

#https://github.com/huawei-noah/ghostnet/blob/master/pytorch/ghostnet.py
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]