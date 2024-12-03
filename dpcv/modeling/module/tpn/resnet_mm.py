import logging
import torch.nn as nn
import torch.utils.checkpoint as cp
from dpcv.checkpoint.load import load_checkpoint
import torch


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 tsm=False,
                 tsm_position='inplace',
                 nsegments=8,
                 groups=1,
                 width_per_group=64,
                 style='pytorch',
                 with_cp=False):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert tsm_position in ['inplace', 'residual']
        self.inplanes = inplanes
        self.planes = planes
        self.tsm = tsm
        self.tsm_position = tsm_position
        self.nsegments = nsegments
        width = int(planes * (width_per_group / 64.)) * groups
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = nn.Conv2d(
            inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=self.conv2_stride,
            groups=groups,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.bn1 = nn.BatchNorm2d(width)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            if self.tsm:
                nt, c, h, w = x.size()
                n_batch = nt // self.nsegments
                x = x.view(n_batch, self.nsegments, c, h, w)
                fold = c // 8
                out = torch.zeros_like(x)
                out[:, :-1, :fold] = x[:, 1:, :fold]
                out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
                out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
                x = out.view(nt, c, h, w)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                if self.tsm_position == 'inplace':
                    identity = self.downsample(x)
                else:
                    identity = self.downsample(identity)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   tsm=False,
                   tsm_position='inplace',
                   nsegments=8,
                   style='pytorch',
                   groups=1,
                   width_per_group=64,
                   with_cp=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            tsm=tsm,
            tsm_position=tsm_position,
            groups=groups,
            width_per_group=width_per_group,
            nsegments=nsegments,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation, style=style, with_cp=with_cp, tsm=tsm, tsm_position=tsm_position,
                  nsegments=nsegments, groups=groups, width_per_group=width_per_group))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNe(x)t backbone.

    Args:
        depth (int): Depth of resnet, from {50, 101}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        partial_bn (bool): Whether to freeze weight and bias of **all but the first** BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 num_stages=4,
                 groups=1,
                 width_per_group=64,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 width=1,
                 frozen_stages=-1,
                 bn_eval=False,
                 bn_frozen=False,
                 partial_bn=False,
                 tsm=False,
                 tsm_position='inplace',
                 nsegments=8,
                 with_cp=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.num_stages = num_stages
        self.tsm = tsm
        self.nsegments = nsegments
        assert 1 <= num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = int(64 * width)
        self.base = int(64 * width)
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = self.base * 2 ** i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                tsm=self.tsm,
                tsm_position=tsm_position,
                nsegments=self.nsegments,
                groups=groups,
                width_per_group=width_per_group,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2 ** (
                len(self.stage_blocks) - 1)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
