import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks import NonLocal2d

from ..builder import NECKS
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS
class FPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 backbone = None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.backbone = None
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # print("=======================")
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

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
@NECKS.register_module()
class NFPN(FPN):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    the paper `Libra R-CNN: Towards Balanced Learning for Object Detection
    <https://arxiv.org/abs/1904.02701>`_ for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_levels,
                 num_outs,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 backbone = None):
        super(NFPN, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = 256
        self.out_channels = 256
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # self.backbone = None
        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels
        self.to_reslayer = nn.ModuleList()
        for out_channel in [64,256,512,1024]:
            d_conv = ConvModule(
                self.in_channels,
                out_channel,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.to_reslayer.append(d_conv)
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()# 64 256 512 1024
        self.fpn = FPN()
        # self.bn = nn.BatchNorm2d()
        for i in range(5):
            d_conv = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            pafpn_conv = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
        self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.refine2 = ConvModule(
                self.in_channels,
                256,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)  
        self.refine_ = ConvModule(
                self.in_channels,
                256,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)   
        # self.gap =SELayer(256)
        # self.gap2 =SELayer(256)
        # self.refine = NonLocal2d(
        #     self.in_channels,
        #     reduction=1,
        #     use_scale=False,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg)        
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    def forward1(self, inputs):
        """Forward function."""
        x = inputs
        inputs = self.fpn(inputs)
        assert len(inputs) == self.num_levels
        # context = context
        # step 1: gather multi-level features by resize and average
        feats = []
        gather_sizes = []
        for i in range(len(inputs)):
            gather_size = inputs[i].size()[2:]
            gather_sizes.append(gather_size)
        # level 0
        # gathered0 = self.gap(self.pafpn_convs[0](F.interpolate(inputs[1], size=gather_sizes[0], mode='bilinear'))) + inputs[0]
        gathered0 = self.pafpn_convs[0](F.interpolate(inputs[1], size=gather_sizes[0], mode='bilinear')) + inputs[0]
        # gathered0 = self.refine(gathered0)
        # level 1
        # gathered1 = self.gap(self.pafpn_convs[1](F.interpolate(inputs[2], size=gather_sizes[1], mode='bilinear'))) + inputs[1] + self.gap2(self.downsample_convs[0](gathered0))
        gathered1 = self.pafpn_convs[1](F.interpolate(inputs[2], size=gather_sizes[1], mode='bilinear')) + inputs[1] + self.downsample_convs[0](gathered0)
        # gathered1 = self.refine(gathered1)
        # level 2_1
        # gathered2_1 = inputs[2] + self.gap2(self.downsample_convs[1](gathered1))
        gathered2_1 = inputs[2] + self.downsample_convs[1](gathered1)
        # level 4
        # gathered4 = inputs[4] + self.gap2(self.pafpn_convs[4](self.downsample_convs[3](inputs[3])))
        gathered4 = inputs[4] + self.pafpn_convs[4](self.downsample_convs[3](inputs[3]))
        # gathered4 = self.pafpn_convs[4](gathered4)        
        # level 3
        gathered3 = self.pafpn_convs[3](F.interpolate(gathered4, size=gather_sizes[3], mode='bilinear')) + inputs[3] + self.downsample_convs[2](gathered2_1)
        # gathered3 = self.gap(self.pafpn_convs[3](F.interpolate(gathered4, size=gather_sizes[3], mode='bilinear'))) + inputs[3] + self.gap2(self.downsample_convs[2](gathered2_1))
        # gathered3 = self.pafpn_convs[3](gathered3) + inputs[3]
        # level 2_2
        # gathered2 = self.gap(self.pafpn_convs[2](F.interpolate(gathered3, size=gather_sizes[2], mode='bilinear'))) +   gathered2_1
        gathered2 = self.pafpn_convs[2](F.interpolate(gathered3, size=gather_sizes[2], mode='bilinear')) +   gathered2_1
        gathered2 =self.refine2(gathered2)
        feats = [gathered0, gathered1, gathered2, gathered3, gathered4]     
        outs = []
        for i in range(self.num_levels):
            out_size = feats[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(gathered2, size=out_size, mode='bilinear')
            else:
                residual = F.adaptive_max_pool2d(gathered2, output_size=out_size)
            residual = self.refine_(residual)
            outs.append(residual + feats[i])        
        # gathered2_1 =             
        # gathered2 = self.pafpn_convs[2](gathered2)

        # level 0 256->64->256->256 s=1
        # bsf = self.refine(bsf)nvidigithub
        # step 3: scatter refined features to multi-levels by a residual path
        # outs = [out_0,out_1,out_2,out_3,out_4]
        return [inputs,tuple(outs)]

    def forward(self, inputs, context):
        """Forward function."""
        x = inputs
        inputs = self.fpn(inputs)
        assert len(inputs) == self.num_levels
        context = context
        # step 1: gather multi-level features by resize and average
        feats = []
        gather_sizes = []
        for i in range(len(inputs)):
            gather_size = inputs[i].size()[2:]
            gather_sizes.append(gather_size)
        # level 0
        # gathered0 = self.gap(self.pafpn_convs[0](F.interpolate(inputs[1], size=gather_sizes[0], mode='bilinear'))) + inputs[0]
        gathered0 = self.pafpn_convs[0](F.interpolate(inputs[1], size=gather_sizes[0], mode='bilinear')) + inputs[0]
        # gathered0 = self.refine(gathered0)
        # level 1
        # gathered1 = self.gap(self.pafpn_convs[1](F.interpolate(inputs[2], size=gather_sizes[1], mode='bilinear'))) + inputs[1] + self.gap2(self.downsample_convs[0](gathered0))
        gathered1 = self.pafpn_convs[1](F.interpolate(inputs[2], size=gather_sizes[1], mode='bilinear')) + inputs[1] + self.downsample_convs[0](gathered0)
        # gathered1 = self.refine(gathered1)
        # level 2_1
        # gathered2_1 = inputs[2] + self.gap2(self.downsample_convs[1](gathered1))
        gathered2_1 = inputs[2] + self.downsample_convs[1](gathered1)
        # level 4
        # gathered4 = inputs[4] + self.gap2(self.pafpn_convs[4](self.downsample_convs[3](inputs[3])))
        gathered4 = inputs[4] + self.pafpn_convs[4](self.downsample_convs[3](inputs[3]))
        # gathered4 = self.pafpn_convs[4](gathered4)        
        # level 3
        # gathered3 = self.gap(self.pafpn_convs[3](F.interpolate(gathered4, size=gather_sizes[3], mode='bilinear'))) + inputs[3] + self.gap2(self.downsample_convs[2](gathered2_1))
        gathered3 = self.pafpn_convs[3](F.interpolate(gathered4, size=gather_sizes[3], mode='bilinear')) + inputs[3] + self.downsample_convs[2](inputs[2])
        # gathered3 = self.pafpn_convs[3](gathered3) + inputs[3]
        # level 2_2
        # gathered2 = self.gap(self.pafpn_convs[2](F.interpolate(gathered3, size=gather_sizes[2], mode='bilinear'))) +   gathered2_1
        gathered2 = self.pafpn_convs[2](F.interpolate(gathered3, size=gather_sizes[2], mode='bilinear')) +   gathered2_1
        # gathered2 = self.pafpn_convs[2](gathered2)
        feats = [gathered0, gathered1, gathered2, gathered3, gathered4]
        for i in range(len(feats)):
        	feats[i] = self.refine(feats[i])
        res_layers = []
        for i, layer_name in enumerate(context.res_layers):
            res_layer = getattr(context, layer_name)
            res_layers.append(res_layer)
        # level 0 256->64->256->256 s=1
        out_0 =feats[0] + inputs[0] + self.fpn.lateral_convs[0](res_layers[0](self.to_reslayer[0](feats[0] + inputs[0]))) + self.refine2(F.interpolate(feats[2], size=gather_sizes[0], mode='bilinear'))
        # out_0 = gathered0 + inputs[0] + self.refine2(F.interpolate(gathered2, size=gather_sizes[0], mode='bilinear'))
        # level 1 256->256->512->256 s=2
        # print("feats[1]:",feats[1].size())
        # print("inputs[1]:",inputs[1].size())
        # print("self.fpn.lateral_convs[1](res_layers[1](self.to_reslayer[1](feats[1] + inputs[1]))):",self.fpn.lateral_convs[1](res_layers[1](self.to_reslayer[1](out_1))).size())
        # print("self.refine2(F.interpolate(feats[2], size=gather_sizes[1], mode='bilinear')):",self.refine2(F.interpolate(feats[2], size=gather_sizes[1], mode='bilinear')).size())
        out_1 =feats[1] + inputs[1] + self.fpn.lateral_convs[1](res_layers[1](self.to_reslayer[1](out_0))) + self.refine2(F.interpolate(feats[2], size=gather_sizes[1], mode='bilinear'))
        # out_1 = gathered1 + inputs[1] + self.refine2(F.interpolate(gathered2, size=gather_sizes[1], mode='nearest'))
        # level 2 256->512->1024->256 s=2
        out_2 =feats[2] + inputs[2] + self.fpn.lateral_convs[2](res_layers[2](self.to_reslayer[2](out_1))) + self.refine2(feats[2])
        # out_2 = gathered2 + inputs[2] 
        # level 3 256->1024->2048->256 s=2
        out_3 =feats[3] + inputs[3] + self.fpn.lateral_convs[3](res_layers[3](self.to_reslayer[3](out_2))) + self.refine2(F.adaptive_max_pool2d(feats[2], output_size=gather_sizes[3]))
        # out_3 = gathered3 + inputs[3] + F.adaptive_max_pool2d(gathered2, output_size=gather_sizes[3])
        # level 3
        out_4 = feats[4] + inputs[4] + self.refine2(F.adaptive_max_pool2d(feats[2], output_size=gather_sizes[4]))
       
        # bsf = self.refine2(bsf)
        # step 3: scatter refined features to multi-levels by a residual path
        outs = [out_0,out_1,out_2,out_3,out_4]
        # outs = [out_0,out_1,out_2,out_3,out_4]
        return [inputs,tuple(outs)]
