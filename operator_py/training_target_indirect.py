import mxnet as mx
import numpy as np



class TrainingTargets(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, overlap_threshold, negative_mining_ratio, negative_mining_thresh, variances):
        super(TrainingTargets, self).__init__()
        self.overlap_threshold = overlap_threshold
        self.negative_mining_ratio = negative_mining_ratio
        self.negative_mining_thresh = negative_mining_thresh
        self.variances = variances

        self.eps = 1e-14

    def forward(self, is_train, req, in_data, out_data, aux):

        anchors = in_data[0]    # [1, num_all_anchors, 4]
        class_preds = in_data[1]    # [batchsize, num_class, num_all_anchors]
        loc_preds = in_data[2]      # [batchsize, 4 x num_all_anchors]
        labels = in_data[3]     # [batchsize, 8, 40]

        box_target, box_mask, cls_target = mx.nd.contrib.MultiBoxTarget(anchors, labels, class_preds,
                                                                            overlap_threshold=self.overlap_threshold,
                                                                            ignore_label=-1,
                                                                            negative_mining_ratio=self.negative_mining_ratio,
                                                                            minimum_negative_samples=0,
                                                                            negative_mining_thresh=self.negative_mining_thresh,
                                                                            variances=self.variances,
                                                                            name="multibox_target")

        anchor_mask = box_mask.reshape(shape=(0, -1, 4))  # [batchsize, num_all_anchors, 4]
        bb8_mask = mx.nd.repeat(data=anchor_mask, repeats=4, axis=2)  # [batchsize, num_all_anchors, 16]

        anchors_in_use = mx.nd.broadcast_mul(lhs=anchor_mask, rhs=anchors)  # [batchsize, num_all_anchors, 4]

        # transform the anchors from [xmin, ymin, xmax, ymax] to [cx, cy, wx, hy]
        centerx = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1) +
                   mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3)) / 2
        centery = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2) +
                   mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4)) / 2
        width = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3) -
                 mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1)) + 1e-8
        height = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4) -
                  mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2)) + 1e-8
        # anchors_in_use_transformed = mx.nd.concat(centerx, centery, width, height, dim=2)   # [batchsize, num_all_anchors, 4]

        # calculate 2D detection results
        loc_preds = loc_preds.reshape(shape=(0, -1, 4)) # [batchsize, num_all_anchors, 4]
        det_center_x = mx.nd.slice_axis(data=loc_preds, axis=2, begin=0, end=1) * self.variances[0] * width + centerx
        det_center_y = mx.nd.slice_axis(data=loc_preds, axis=2, begin=1, end=2) * self.variances[1] * height + centery
        det_width = mx.nd.exp(data=mx.nd.slice_axis(data=loc_preds, axis=2, begin=2, end=3) * self.variances[2]) * width
        det_height = mx.nd.exp(data=mx.nd.slice_axis(data=loc_preds, axis=2, begin=3, end=4) * self.variances[3]) * height
        det_results_transformed = mx.nd.concat(det_center_x, det_center_y, det_width, det_height, dim=2)    # [batchsize, num_all_anchors, 4]

        bb8_target = mx.nd.zeros_like(data=bb8_mask)    # batchsize x num_anchors x 16
        bb8_label = mx.nd.slice_axis(data=labels, axis=2, begin=8, end=24)

        # calculate targets for OCCLUSION dataset
        for cid in range(1, 9):
            cid_target_mask = (cls_target == cid)
            cid_target_mask = cid_target_mask.reshape(shape=(0,-1,1))
            # cid_anchors_in_use_transformed = mx.nd.broadcast_mul(lhs=cid_target_mask, rhs=anchors_in_use_transformed)
            cid_anchors_in_use_transformed = mx.nd.where(condition=mx.nd.broadcast_to(cid_target_mask, shape=det_results_transformed.shape),
                                                        x=det_results_transformed,
                                                        y=mx.nd.zeros_like(det_results_transformed))
            cid_label_mask = (mx.nd.slice_axis(data=labels, axis=2, begin=0, end=1) == cid - 1)
            cid_bb8_label = mx.nd.broadcast_mul(lhs=cid_label_mask, rhs=bb8_label)
            cid_bb8_label = mx.nd.max(cid_bb8_label, axis=1, keepdims=True) # batchsize x 1 x 16

            # substract center
            cid_bb8_target = mx.nd.broadcast_sub(cid_bb8_label, mx.nd.tile(  # repeat single element !! error
                data=mx.nd.slice_axis(cid_anchors_in_use_transformed, axis=2, begin=0, end=2),
                reps=(1, 1, 8)))
            # divide by w and h
            cid_bb8_target = mx.nd.broadcast_div(cid_bb8_target, mx.nd.tile(
                data=mx.nd.slice_axis(cid_anchors_in_use_transformed, axis=2, begin=2, end=4),
                reps=(1, 1, 8))) / 0.1  # variance

            cid_bb8_target = mx.nd.where(condition=mx.nd.broadcast_to(cid_target_mask, shape=cid_bb8_target.shape),
                                         x=cid_bb8_target,
                                         y=mx.nd.zeros_like(cid_bb8_target))
            bb8_target = bb8_target + cid_bb8_target

        condition = bb8_mask > 0.5
        bb8_target = mx.nd.where(condition=condition, x=bb8_target, y=mx.nd.zeros_like(data=bb8_target))

        bb8_target = bb8_target.flatten()  # batchsize x (num_anchors x 16)
        bb8_mask = bb8_mask.flatten()  # batchsize x (num_anchors x 16)

        self.assign(out_data[0], req[0], box_target)
        self.assign(out_data[1], req[1], box_mask)
        self.assign(out_data[2], req[2], cls_target)
        self.assign(out_data[3], req[3], bb8_target)
        self.assign(out_data[4], req[4], bb8_mask)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("training_targets_indirect")
class TrainingTargetsProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, overlap_threshold=0.5, negative_mining_ratio=3,
                 negative_mining_thresh=0.5, variances=(0.1, 0.1, 0.2, 0.2)):
        #
        super(TrainingTargetsProp, self).__init__(need_top_grad=False)
        self.overlap_threshold = float(overlap_threshold)
        self.negative_mining_ratio = float(negative_mining_ratio)
        self.negative_mining_thresh = float(negative_mining_thresh)
        self.variances = variances

    def list_arguments(self):
        return ['anchors', 'cls_preds', 'loc_preds', 'labels']

    def list_outputs(self):
        return ['box_target', 'box_mask', 'cls_target', 'bb8_target', 'bb8_mask']

    def infer_shape(self, in_shape):
        anchors_shape = in_shape[0]
        data_shape = in_shape[1]    # [batchsize, num_classes, num_all_anchors]
        loc_preds_shape = in_shape[2]
        label_shape = in_shape[3]

        box_target_shape = (data_shape[0], 4 * data_shape[2])
        box_mask_shape = (data_shape[0], 4 * data_shape[2])
        cls_target_shape = (data_shape[0], data_shape[2])
        bb8_target_shape = (data_shape[0], 16 * data_shape[2])
        bb8_mask_shape = (data_shape[0], 16 * data_shape[2])

        return [anchors_shape, data_shape, loc_preds_shape, label_shape], \
               [box_target_shape, box_mask_shape,
                cls_target_shape, bb8_target_shape, bb8_mask_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return TrainingTargets(self.overlap_threshold, self.negative_mining_ratio, self.negative_mining_thresh, self.variances)
