import mxnet as mx

eps = 2e-5
use_global_stats = True
workspace = 512
vgg_deps = {'16': (2, 2, 3, 3, 3, 1, 1)}
units = vgg_deps['16']
filter_list = [64, 128, 256, 512, 512, 1024, 1024]

def conv_layer(data, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type=None, use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    bias = mx.symbol.Variable(name="{}_conv_bias".format(name),
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})
    conv = mx.symbol.Convolution(data=data, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="{}_conv".format(name), bias=bias)
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="{}_bn".format(name))
    if act_type is not None:
        conv = mx.symbol.Activation(data=conv, act_type=act_type, \
            name="{}_{}".format(name, act_type))
    return conv


def get_vgg_reduced_conv(data, num_layers):
    units = vgg_deps[str(int(num_layers))]
    # group 1
    unit = data
    for i in range(1, units[0] + 1):
        unit = mx.symbol.Convolution(data=unit, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[0], name="conv1_{}".format(i))
        unit = mx.symbol.Activation(data=unit, act_type="relu", name="relu1_{}".format(i))
    unit = mx.symbol.Pooling(
        data=unit, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    for i in range(1, units[1] + 1):
        unit = mx.symbol.Convolution(data=unit, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[1], name="conv2_{}".format(i))
        unit = mx.symbol.Activation(data=unit, act_type="relu", name="relu2_{}".format(i))
    relu2_2 = unit
    unit = mx.symbol.Pooling(
        data=unit, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    for i in range(1, units[2] + 1):
        unit = mx.symbol.Convolution(data=unit, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[2], name="conv3_{}".format(i))
        unit = mx.symbol.Activation(data=unit, act_type="relu", name="relu3_{}".format(i))
    relu3_3 = unit  # stride 4
    unit = mx.symbol.Pooling(
        data=unit, pool_type="max", kernel=(2, 2), stride=(2, 2), pooling_convention="full", name="pool3")
    # group 4
    for i in range(1, units[3] + 1):
        unit = mx.symbol.Convolution(data=unit, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[3], name="conv4_{}".format(i))
        unit = mx.symbol.Activation(data=unit, act_type="relu", name="relu4_{}".format(i))
    relu4_3 = unit      # stride 8
    unit = mx.symbol.Pooling(
        data=unit, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    for i in range(1, units[4] + 1):
        unit = mx.symbol.Convolution(data=unit, kernel=(3, 3), pad=(1, 1), num_filter=filter_list[4], name="conv5_{}".format(i))
        unit = mx.symbol.Activation(data=unit, act_type="relu", name="relu5_{}".format(i))
    relu5_3 = unit
    unit = mx.symbol.Pooling(
        data=unit, pool_type="max", kernel=(2, 2), stride=(1, 1), name="pool5")
    # group 6
    conv6 = mx.symbol.Convolution(
        data=unit, kernel=(3, 3), pad=(6, 6), dilate=(6, 6),
        num_filter=filter_list[5], name="fc6")
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    conv7 = mx.symbol.Convolution(
        data=relu6, kernel=(1, 1), pad=(0, 0), num_filter=filter_list[6], name="fc7")
    relu7 = mx.symbol.Activation(data=conv7, act_type="relu", name="relu7")  # stride 16

    return [relu7, relu4_3, relu3_3]


def get_vgg_reduced_conv_down(conv_feat):
    relu7, relu4_3, relu3_3 = conv_feat
    # C5 to P5, 1x1 dimension reduction to 256
    P5 = conv_layer(data=relu7, kernel=(1, 1), num_filter=256, name="P5_lateral")
    P5_up = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    P5 = conv_layer(data=P5, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P5")

    # P5 2x upsampling + C4 = P4
    P4_la = conv_layer(data=relu4_3, kernel=(1, 1), num_filter=256, name="P4_lateral")
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4 = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    P4_up = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P4 = conv_layer(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4")

    # P4 2x upsampling + C3 = P3
    P3_la = conv_layer(data=relu3_3, kernel=(1, 1), num_filter=256, name="P3_lateral")
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3 = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    # P3_up = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    P3 = conv_layer(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3")

    # P6 2x subsampling P5
    P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6')

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride32": P6, "stride16": P5, "stride8": P4, "stride4": P3})

    return conv_fpn_feat, [P6, P5, P4, P3]

def get_symbol(num_classes=1000, **kwargs):
    """
    VGG 16 layers network
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")   # stride 2
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")   # stride 4
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), \
        pooling_convention="full", name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")   # stride 8
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(3, 3), stride=(1, 1),
        pad=(1,1), name="pool5")
    # group 6
    conv6 = mx.symbol.Convolution(
        data=pool5, kernel=(3, 3), pad=(6, 6), dilate=(6, 6),
        num_filter=1024, name="fc6")
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    conv7 = mx.symbol.Convolution(
        data=relu6, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc7")
    relu7 = mx.symbol.Activation(data=conv7, act_type="relu", name="relu7")     # stride 16
    # drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")

    gpool = mx.symbol.Pooling(data=relu7, pool_type='avg', kernel=(7, 7),
        global_pool=True, name='global_pool')
    conv8 = mx.symbol.Convolution(data=gpool, num_filter=num_classes, kernel=(1, 1),
        name='fc8')
    flat = mx.symbol.Flatten(data=conv8)
    softmax = mx.symbol.SoftmaxOutput(data=flat, name='softmax')
    return softmax
