'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import mxnet as mx
from symbol.common import conv_act_layer

eps = 2e-5
use_global_stats = True
workspace = 512
res_deps = {'50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
units = res_deps['50']
filter_list = [256, 512, 1024, 2048]


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, dilate=(1, 1), bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=dilate,
                                   dilate=dilate,
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=dilate,
                                   dilate=dilate,
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=dilate,
                                   dilate=dilate,
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut


def get_resnet_conv(data, num_layers):
    units = res_deps[str(int(num_layers))]
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0   = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0   = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True,
                             name='stage1_unit%s' % i)
    conv_C2 = unit

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True,
                             name='stage2_unit%s' % i)
    conv_C3 = unit

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True,
                             name='stage3_unit%s' % i)
    conv_C4 = unit

    # res5
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                             name='stage4_unit%s' % i)
    conv_C5 = unit

    conv_feat = [conv_C5, conv_C4, conv_C3, conv_C2]
    return conv_feat


def get_resnet_conv_down(conv_feat):
    # C5 to P5, 1x1 dimension reduction to 256
    C5 = conv_feat[0]
    P5 = conv_act_layer(from_layer=C5, kernel=(1, 1), num_filter=256, name="P5_lateral", use_act=False)
    P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    P5 = conv_act_layer(from_layer=P5, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P5", use_act=False)

    # P5 2x upsampling + C4 = P4
    P4_la   = conv_act_layer(from_layer=conv_feat[1], kernel=(1, 1), num_filter=256, name="P4_lateral", use_act=False)
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4      = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    P4_up = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P4      = conv_act_layer(from_layer=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4", use_act=False)

    # P4 2x upsampling + C3 = P3
    P3_la   = conv_act_layer(from_layer=conv_feat[2], kernel=(1, 1), num_filter=256, name="P3_lateral", use_act=False)
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3      = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    P3_up = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    P3      = conv_act_layer(from_layer=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3", use_act=False)

    # P3 2x upsampling + C2 = P2
    P2_la   = conv_act_layer(from_layer=conv_feat[3], kernel=(1, 1), num_filter=256, name="P2_lateral", use_act=False)
    P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
    P2      = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
    P2      = conv_act_layer(from_layer=P2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2", use_act=False)

    # P6 2x subsampling P5
    P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6')

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride64":P6, "stride32":P5, "stride16":P4, "stride8":P3, "stride4":P2})

    return conv_fpn_feat, [P6, P5, P4, P3, P2]


def get_resnet_conv_down_mask_style(conv_feat):
    # C5 to P5, 1x1 dimension reduction to 256
    P5 = mx.symbol.Convolution(data=conv_feat[0], kernel=(1, 1), num_filter=256, name="P5_lateral")

    # P5 2x upsampling + C4 = P4
    P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    P4_la = mx.symbol.Convolution(data=conv_feat[1], kernel=(1, 1), num_filter=256, name="P4_lateral")
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4 = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    P4 = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4_aggregate")

    # P4 2x upsampling + C3 = P3
    P4_up = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P3_la = mx.symbol.Convolution(data=conv_feat[2], kernel=(1, 1), num_filter=256, name="P3_lateral")
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3 = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    P3 = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_aggregate")

    # P3 2x upsampling + C2 = P2
    P3_up = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    P2_la = mx.symbol.Convolution(data=conv_feat[3], kernel=(1, 1), num_filter=256, name="P2_lateral")
    P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
    P2 = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
    P2 = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2_aggregate")

    # P6 2x subsampling P5
    P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6_subsampling')

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride64": P6, "stride32": P5, "stride16": P4, "stride8": P3, "stride4": P2})

    return conv_fpn_feat, [P6, P5, P4, P3, P2]


def get_ssd_conv(data, num_layers):
    conv_C5, conv_C4, conv_C3, _ = get_resnet_conv(data, num_layers)

    # extra conv C6
    conv_1x1 = conv_act_layer(conv_C5, 'multi_feat_2_conv_1x1',
                              128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    conv_C6 = conv_act_layer(conv_1x1, 'multi_feat_2_conv_3x3',
                              256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')

    # extra conv C7
    conv_1x1 = conv_act_layer(conv_C6, 'multi_feat_3_conv_1x1',
                              128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    conv_C7 = conv_act_layer(conv_1x1, 'multi_feat_3_conv_3x3',
                             256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')

    # extra conv C8
    conv_1x1 = conv_act_layer(conv_C7, 'multi_feat_4_conv_1x1',
                              128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    conv_C8 = conv_act_layer(conv_1x1, 'multi_feat_4_conv_3x3',
                             256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')

    # extra conv C9
    conv_1x1 = conv_act_layer(conv_C8, 'multi_feat_5_conv_1x1',
                              128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    conv_C9 = conv_act_layer(conv_1x1, 'multi_feat_5_conv_3x3',
                             256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')

    conv_feat = [conv_C7, conv_C6, conv_C5, conv_C4, conv_C3]
    return conv_feat


def get_ssd_conv_down(conv_feat):
    conv_C7, conv_C6, conv_C5, conv_C4, conv_C3 = conv_feat

    # # C6 to P6, 1x1 dimension reduction to 256
    # P6 = conv_act_layer(from_layer=conv_C6, kernel=(1, 1), num_filter=256, name="P6_lateral", use_act=False)
    # P6_up = mx.symbol.UpSampling(P6, scale=2, sample_type='nearest', workspace=512, name='P6_upsampling', num_args=1)
    # P6 = conv_act_layer(from_layer=P6, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P6", use_act=False)
    #
    # # P6 2x upsampling + C5 = P5
    # P5_la = conv_act_layer(from_layer=conv_C5, kernel=(1, 1), num_filter=256, name="P5_lateral", use_act=False)
    # P6_clip = mx.symbol.Crop(*[P6_up, P5_la], name="P5_clip")
    # P5 = mx.sym.ElementWiseSum(*[P6_clip, P5_la], name="P5_sum")
    # P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    # P5 = conv_act_layer(from_layer=P5, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P5", use_act=False)
    #
    # # P5 2x upsampling + C4 = P4
    # P4_la = conv_act_layer(from_layer=conv_C4, kernel=(1, 1), num_filter=256, name="P4_lateral", use_act=False)
    # P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    # P4 = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    # P4 = conv_act_layer(from_layer=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4", use_act=False)
    #
    # conv_fpn_feat = dict()
    # conv_fpn_feat.update({"stride256": conv_C8, "stride128": conv_C7, "stride64": P6, "stride32": P5, "stride16": P4})
    #
    # return conv_fpn_feat, [conv_C8, conv_C7, P6, P5, P4]

    # C5 to P5, 1x1 dimension reduction to 256
    P5 = conv_act_layer(from_layer=conv_C5, kernel=(1, 1), num_filter=256, name="P5_lateral", use_act=False)
    P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    P5 = conv_act_layer(from_layer=P5, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P5", use_act=False)

    # P5 2x upsampling + C4 = P4
    P4_la = conv_act_layer(from_layer=conv_C4, kernel=(1, 1), num_filter=256, name="P4_lateral", use_act=False)
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4 = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    P4_up = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P4 = conv_act_layer(from_layer=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4", use_act=False)

    # P4 2x upsampling + C3 = P3
    P3_la = conv_act_layer(from_layer=conv_C3, kernel=(1, 1), num_filter=256, name="P3_lateral", use_act=False)
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3 = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    P3 = conv_act_layer(from_layer=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3", use_act=False)

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride128": conv_C7, "stride64": conv_C6, "stride32": P5, "stride16": P4, "stride8": P3})

    return conv_fpn_feat, [conv_C7, conv_C6, P5, P4, P3]


def get_detnet_conv(data, num_layers):
    _, _, conv_C3, conv_C2 = get_resnet_conv(data, num_layers)
    #  detnet res4 stride 8
    unit = residual_unit(data=conv_C3, num_filter=1024, stride=(1, 1), dim_match=False, name='stage3_unit1', dilate=(2,2))
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=1024, stride=(1, 1), dim_match=True,
                             name='stage3_unit%s' % i)
    conv_C4 = unit
    # detnet res5 stride 8
    unit = residual_unit(data=unit, num_filter=2048, stride=(1, 1), dim_match=False, name='stage4_unit1', dilate=(2,2))
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=2048, stride=(1, 1), dim_match=True,
                             name='stage4_unit%s' % i)
    conv_C5 = unit

    # extra conv C6 stride 16
    conv_1x1 = conv_act_layer(unit, 'multi_feat_3_conv_1x1',
                              128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    conv_C6 = conv_act_layer(conv_1x1, 'multi_feat_3_conv_3x3',
                             256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')

    # extra conv C7 stride 32
    conv_1x1 = conv_act_layer(conv_C6, 'multi_feat_4_conv_1x1',
                              128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    conv_C7 = conv_act_layer(conv_1x1, 'multi_feat_4_conv_3x3',
                             256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')

    conv_feat = [conv_C7, conv_C6, conv_C5, conv_C4, conv_C3]

    # _, conv_C4, conv_C3, conv_C2 = get_resnet_conv(data, num_layers)
    #
    # # detnet res5 stride 16
    # unit = residual_unit(data=conv_C4, num_filter=2048, stride=(1, 1), dim_match=False, name='stage4_unit1', dilate=(2, 2))
    # for i in range(2, units[3] + 1):
    #     unit = residual_unit(data=unit, num_filter=2048, stride=(1, 1), dim_match=True,
    #                          name='stage4_unit%s' % i)
    # conv_C5 = unit
    #
    # # extra conv C6 stride 16
    # conv_1x1 = conv_act_layer(unit, 'multi_feat_2_conv_1x1',
    #                           256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    # conv_C6 = conv_act_layer(conv_1x1, 'multi_feat_2_conv_3x3',
    #                          512, kernel=(3, 3), pad=(2, 2), stride=(1, 1), dilate=(2, 2), act_type='relu')
    #
    # # extra conv C7 stride 32
    # conv_1x1 = conv_act_layer(conv_C6, 'multi_feat_3_conv_1x1',
    #                           128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    # conv_C7 = conv_act_layer(conv_1x1, 'multi_feat_3_conv_3x3',
    #                          256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')
    #
    # # extra conv C8 stride 64
    # conv_1x1 = conv_act_layer(conv_C7, 'multi_feat_4_conv_1x1',
    #                           128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    # conv_C8 = conv_act_layer(conv_1x1, 'multi_feat_4_conv_3x3',
    #                          256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')
    #
    # conv_feat = [conv_C8, conv_C7, conv_C6, conv_C5, conv_C4]
    return conv_feat


def get_deeplabv2_conv(data, num_layers):
    _, _, conv_C3, conv_C2 = get_resnet_conv(data, num_layers)
    #  deeplabv2 res4 stride 8
    unit = residual_unit(data=conv_C3, num_filter=1024, stride=(1, 1), dim_match=False, name='stage3_unit1', dilate=(2,2))
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=1024, stride=(1, 1), dim_match=True, dilate=(2, 2),
                             name='stage3_unit%s' % i)
    conv_C4 = unit
    # deeplabv2 res5 stride 8
    unit = residual_unit(data=unit, num_filter=2048, stride=(1, 1), dim_match=False, name='stage4_unit1', dilate=(4, 4))
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=2048, stride=(1, 1), dim_match=True, dilate=(4, 4),
                             name='stage4_unit%s' % i)
    conv_C5 = unit

    # extra conv C6 stride 16
    conv_1x1 = conv_act_layer(unit, 'multi_feat_3_conv_1x1',
                              128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    conv_C6 = conv_act_layer(conv_1x1, 'multi_feat_3_conv_3x3',
                             256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')

    # extra conv C7 stride 32
    conv_1x1 = conv_act_layer(conv_C6, 'multi_feat_4_conv_1x1',
                              128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    conv_C7 = conv_act_layer(conv_1x1, 'multi_feat_4_conv_3x3',
                             256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')

    conv_feat = [conv_C7, conv_C6, conv_C5, conv_C4, conv_C3]

    # _, conv_C4, conv_C3, conv_C2 = get_resnet_conv(data, num_layers)
    #
    # # detnet res5 stride 16
    # unit = residual_unit(data=conv_C4, num_filter=2048, stride=(1, 1), dim_match=False, name='stage4_unit1', dilate=(2, 2))
    # for i in range(2, units[3] + 1):
    #     unit = residual_unit(data=unit, num_filter=2048, stride=(1, 1), dim_match=True,
    #                          name='stage4_unit%s' % i)
    # conv_C5 = unit
    #
    # # extra conv C6 stride 16
    # conv_1x1 = conv_act_layer(unit, 'multi_feat_2_conv_1x1',
    #                           256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    # conv_C6 = conv_act_layer(conv_1x1, 'multi_feat_2_conv_3x3',
    #                          512, kernel=(3, 3), pad=(2, 2), stride=(1, 1), dilate=(2, 2), act_type='relu')
    #
    # # extra conv C7 stride 32
    # conv_1x1 = conv_act_layer(conv_C6, 'multi_feat_3_conv_1x1',
    #                           128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    # conv_C7 = conv_act_layer(conv_1x1, 'multi_feat_3_conv_3x3',
    #                          256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')
    #
    # # extra conv C8 stride 64
    # conv_1x1 = conv_act_layer(conv_C7, 'multi_feat_4_conv_1x1',
    #                           128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    # conv_C8 = conv_act_layer(conv_1x1, 'multi_feat_4_conv_3x3',
    #                          256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu')
    #
    # conv_feat = [conv_C8, conv_C7, conv_C6, conv_C5, conv_C4]
    return conv_feat


def get_detnet_conv_down(conv_feat):
    # conv_C8, conv_C7, conv_C6, conv_C5, conv_C4 = conv_feat
    #
    # # C6 to P6, 1x1 dimension reduction to 256
    # P6 = conv_act_layer(from_layer=conv_C6, kernel=(1, 1), num_filter=256, name="P6_lateral", use_act=False)
    # P6_up = mx.symbol.UpSampling(P6, scale=2, sample_type='nearest', workspace=512, name='P6_upsampling', num_args=1)
    # P6 = conv_act_layer(from_layer=P6, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P6", use_act=False)
    #
    # # P6 2x upsampling + C5 = P5
    # P5_la = conv_act_layer(from_layer=conv_C5, kernel=(1, 1), num_filter=256, name="P5_lateral", use_act=False)
    # P6_clip = mx.symbol.Crop(*[P6_up, P5_la], name="P5_clip")
    # P5 = mx.sym.ElementWiseSum(*[P6_clip, P5_la], name="P5_sum")
    # P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    # P5 = conv_act_layer(from_layer=P5, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P5", use_act=False)
    #
    # # P5 2x upsampling + C4 = P4
    # P4_la = conv_act_layer(from_layer=conv_C4, kernel=(1, 1), num_filter=256, name="P4_lateral", use_act=False)
    # P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    # P4 = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    # P4 = conv_act_layer(from_layer=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4", use_act=False)
    #
    # conv_fpn_feat = dict()
    # conv_fpn_feat.update({"stride256": conv_C8, "stride128": conv_C7, "stride64": P6, "stride32": P5, "stride16": P4})
    #
    # return conv_fpn_feat, [conv_C8, conv_C7, P6, P5, P4]
    conv_C7, conv_C6, conv_C5, conv_C4, conv_C3 = conv_feat

    # C5 to P5, 1x1 dimension reduction to 256
    P5 = conv_act_layer(from_layer=conv_C5, kernel=(1, 1), num_filter=256, name="P5_lateral", use_act=False)
    # P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)


    # P5 2x upsampling + C4 = P4
    P4_la = conv_act_layer(from_layer=conv_C4, kernel=(1, 1), num_filter=256, name="P4_lateral", use_act=False)
    # P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4 = mx.sym.ElementWiseSum(*[P5, P4_la], name="P4_sum")
    # P4_up = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)


    # P4 2x upsampling + C3 = P3
    P3_la = conv_act_layer(from_layer=conv_C3, kernel=(1, 1), num_filter=256, name="P3_lateral", use_act=False)
    # P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3 = mx.sym.ElementWiseSum(*[P4, P3_la], name="P3_sum")


    P5 = conv_act_layer(from_layer=P5, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P5", use_act=False)
    P4 = conv_act_layer(from_layer=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4", use_act=False)
    P3 = conv_act_layer(from_layer=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3", use_act=False)

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride64": conv_C7, "stride32": conv_C6, "stride16": P5, "stride8": P4, "stride4": P3})

    return conv_fpn_feat, [conv_C7, conv_C6, P5, P4, P3]


def resnet(units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')


def get_symbol(num_classes, num_layers, image_shape, conv_workspace=256, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return resnet(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)
