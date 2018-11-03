import mxnet as mx
from symbol.common import multi_layer_feature_SSD, multibox_layer_FPN, multibox_layer_SSD
# from operator_py.training_target import *
from operator_py.training_target_indirect import *

def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)


def training_targets(anchors, class_preds, labels):
    # labels_np = labels.asnumpy()
    # view_cls_label = mx.nd.slice_axis(data=labels, axis=2, begin=6, end=7)
    # inplane_cls_label = mx.nd.slice_axis(data=labels, axis=2, begin=7, end=8)
    # bbox_label = mx.nd.slice_axis(data=labels, axis=2, begin=1, end=5)
    # label_valid_count = mx.symbol.sum(mx.symbol.slice_axis(labels, axis=2, begin=0, end=1) >= 0, axis=1)
    # class_preds = class_preds.transpose(axes=(0,2,1))

    box_target, box_mask, cls_target = mx.symbol.contrib.MultiBoxTarget(anchors, labels, class_preds, overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")

    anchor_mask = box_mask.reshape(shape=(0, -1, 4))    # batchsize x num_anchors x 4
    bb8_mask = mx.symbol.repeat(data=anchor_mask, repeats=4, axis=2)  # batchsize x num_anchors x 16
    #anchor_mask = mx.nd.mean(data=anchor_mask, axis=2, keepdims=False, exclude=False)

    anchors_in_use = mx.symbol.broadcast_mul(lhs=anchor_mask,rhs=anchors)   # batchsize x num_anchors x 4

    # transform the anchors from [xmin, ymin, xmax, ymax] to [cx, cy, wx, hy]

    centerx = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1) + \
               mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3)) / 2
    centery = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2) + \
               mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4)) / 2
    width = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3) - \
               mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1)) + 0.0000001
    height = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4) - \
               mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2)) + 0.0000001
    # anchors_in_use_transformed = mx.symbol.zeros_like(data=anchors_in_use)
    # anchors_in_use_transformed[:, :, 0] = (anchors_in_use[:, :, 0] + anchors_in_use[:, :, 2]) / 2
    # anchors_in_use_transformed[:, :, 1] = (anchors_in_use[:, :, 1] + anchors_in_use[:, :, 3]) / 2
    # anchors_in_use_transformed[:, :, 2] = anchors_in_use[:, :, 2] - anchors_in_use[:, :, 0] + 0.0000001
    # anchors_in_use_transformed[:, :, 3] = anchors_in_use[:, :, 3] - anchors_in_use[:, :, 1] + 0.0000001
    anchors_in_use_transformed = mx.symbol.concat(centerx, centery, width, height, dim=2)   # batchsize x num_anchors x 4

    bb8_target = mx.symbol.zeros_like(bb8_mask)
    bb8_label = mx.symbol.slice_axis(data=labels, axis=2, begin=8, end=24)
    # cls_target_temp = mx.symbol.repeat(data=cls_target, repeats=4, axis=1)
    # cls_target_temp = mx.symbol.reshape(data=cls_target_temp, shape=(0, -1, 4)) # batchsize x num_anchors x 4
    # calculate targets for OCCLUSION dataset
    for cid in range(1,9):
        # cid_target_mask = (cls_target == cid)
        # cid_target_mask = mx.symbol.reshape(data=cid_target_mask, shape=(0,-1,1))   # batchsize x num_anchors x 1
        # cid_anchors_in_use_transformed = mx.symbol.broadcast_mul(lhs=cid_target_mask, rhs=anchors_in_use_transformed)   # batchsize x num_anchors x 4
        cid_anchors_in_use_transformed = mx.symbol.where(condition=(cls_target==cid), x=anchors_in_use_transformed,
                                                         y=mx.symbol.zeros_like(anchors_in_use_transformed))
        cid_label_mask = (mx.symbol.slice_axis(data=labels, axis=2, begin=0, end=1) == cid-1)
        cid_bb8_label = mx.symbol.broadcast_mul(lhs=cid_label_mask, rhs=bb8_label)
        cid_bb8_label = mx.symbol.max(cid_bb8_label, axis=1, keepdims=True) # batchsize x 1 x 16

        # substract center
        cid_bb8_target = mx.symbol.broadcast_sub(cid_bb8_label, mx.symbol.tile(   # repeat single element !! error
            data=mx.symbol.slice_axis(cid_anchors_in_use_transformed, axis=2, begin=0, end=2),
            reps=(1,1,8)))
        # divide by w and h
        cid_bb8_target = mx.symbol.broadcast_div(cid_bb8_target, mx.symbol.tile(
            data=mx.symbol.slice_axis(cid_anchors_in_use_transformed, axis=2, begin=2, end=4),
            reps=(1, 1, 8))) / 0.1  # variance
        # cid_bb8_target = mx.symbol.broadcast_mul(lhs=cid_target_mask, rhs=cid_bb8_target)   # this sentence will cause loss explosion, don't know why
        # cid_bb8_target = mx.symbol.where(condition=(mx.symbol.repeat(cls_target_temp, repeats=4, axis=2)==cid), x=cid_bb8_target,
        #                                  y=mx.symbol.zeros_like(cid_bb8_target))
        cid_bb8_target = mx.symbol.where(condition=(cls_target == cid),
                                         x=cid_bb8_target,
                                         y=mx.symbol.zeros_like(cid_bb8_target))
        bb8_target = bb8_target + cid_bb8_target

    condition = bb8_mask > 0.5
    bb8_target = mx.symbol.where(condition=condition, x=bb8_target, y=mx.symbol.zeros_like(data=bb8_target))

    bb8_target = bb8_target.flatten()   # batchsize x (num_anchors x 16)
    bb8_mask = bb8_mask.flatten()       # batchsize x (num_anchors x 16)
    return box_target, box_mask, cls_target, bb8_target, bb8_mask


def get_symbol_train(network, num_classes, alpha_bb8, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, minimum_negative_samples=0, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    label = mx.sym.Variable('label')
    body = import_module(network).get_symbol(num_classes=num_classes, **kwargs)

    layers = multi_layer_feature_SSD(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_SSD(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_resnet_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnet import get_ssd_conv, get_ssd_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_FPN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_resnetd_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnet import get_detnet_conv, get_detnet_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_detnet_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_detnet_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_FPN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_resnetdeeplabv2_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnet import get_deeplabv2_conv, get_detnet_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_deeplabv2_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_detnet_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_FPN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_resnetm_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_FPN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out

def get_resnetm_ssd_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_conv, get_ssd_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_conv(data, num_layers)

    # shared convolutional layers, top down
    # _, conv_fpn_feat = get_ssd_conv_down(conv_feat)
    conv_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_SSD(conv_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets_indirect",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            loc_preds=loc_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out

def get_resnetmd_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.resnetm import get_ssd_md_conv, get_ssd_md_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_ssd_md_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_ssd_md_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6, P7]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer_FPN(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_vgg_reduced_fpn_train(num_classes, alpha_bb8, num_layers, num_filters,
                     sizes, ratios, normalizations=-1, steps=[], **kwargs):
    """Build network symbol for training FPN

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    from symbol.vgg16_reduced import get_vgg_reduced_conv, get_vgg_reduced_conv_down
    data = mx.symbol.Variable('data')
    label = mx.sym.Variable('label')

    # shared convolutional layers, bottom up
    conv_feat = get_vgg_reduced_conv(data, num_layers)

    # shared convolutional layers, top down
    _, conv_fpn_feat = get_vgg_reduced_conv_down(conv_feat)
    conv_fpn_feat.reverse()     # [P3, P4, P5, P6]

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer(conv_fpn_feat, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label)
    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.symbol.Custom(op_type="training_targets",
                                                                                            name="training_targets",
                                                                                            anchors=anchor_boxes,
                                                                                            cls_preds=cls_preds,
                                                                                            labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out


def get_symbol(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    body = import_module(network).get_symbol(num_classes=num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return out
