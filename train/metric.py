import mxnet as mx
import numpy as np
from MultiBoxDetection import myMultiBoxDetection, BB8MultiBoxDetection


class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(MultiBoxMetric, self).__init__('MultiBox')
        self.eps = eps
        self.num = 7
        self.ovp_thresh = 0.5
        self.use_difficult = False
        self.name = ['CrossEntropy', 'loc_SmoothL1', 'loc_MAE', 'loc_MAE_pixel', 'bb8_SmoothL1', 'bb8_MAE', 'bb8_MAE_pixel']
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        :param preds: [cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae]
        Implementation of updating metrics
        """
        labels = labels[0].asnumpy()
        # get generated multi label from network
        cls_prob = preds[0]
        loc_loss = preds[1].asnumpy()     # smoothL1 loss
        # loc_loss_in_use = loc_loss[loc_loss.nonzero()]
        cls_target = preds[2].asnumpy()
        bb8_loss = preds[3].asnumpy()
        loc_pred = preds[4]
        bb8_pred = preds[5]
        anchors = preds[6]
        # anchor_in_use = anchors[anchors.nonzero()]
        bb8dets = BB8MultiBoxDetection(cls_prob, loc_pred, bb8_pred, anchors, nms_threshold=0.5, force_suppress=False,
                                      variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
        bb8dets = bb8dets.asnumpy()

        # monitor results
        # loc_target = preds[7].asnumpy()
        # loc_label_in_use = loc_target[loc_target.nonzero()]
        # loc_pred_masked = preds[8].asnumpy()
        # loc_pred_in_use = loc_pred_masked[loc_pred_masked.nonzero()]
        # bb8_target = preds[10].asnumpy()
        # bb8_label_in_use = bb8_target[bb8_target.nonzero()]
        # bb8_pred_masked = preds[11].asnumpy()
        # bb8_pred_in_use = bb8_pred_masked[bb8_pred_masked.nonzero()]
        loc_mae = preds[9].asnumpy()
        # loc_mae_in_use = loc_mae[loc_mae.nonzero()]
        bb8_mae = preds[12].asnumpy()
        # bb8_mae_in_use = bb8_mae[bb8_mae.nonzero()]

        # for each class, only consider the most confident instance
        loc_mae_pixel = []
        bb8_mae_pixel = []
        image_size = 300.0
        for sampleDet, sampleLabel in zip(bb8dets, labels):
            for instanceLabel in sampleLabel:
                if instanceLabel[0] < 0:
                    continue
                else:
                    cid = instanceLabel[0]
                    indices = np.where(sampleDet[:, 0] == cid)[0]
                    if indices.size > 0:
                        instanceDet = sampleDet[indices[0]] # only consider the most confident instance
                        loc_mae_pixel.append(np.abs((instanceDet[2:6] - instanceLabel[1:5]) * image_size))
                        bb8_mae_pixel.append(np.abs((instanceDet[6:22] - instanceLabel[8:24]) * image_size))
                    # else:
                    #     # punish missed gt
                    #     loc_mae_pixel.append(np.ones(shape=(4,), dtype=np.float32) * image_size)
                    #     bb8_mae_pixel.append(np.ones(shape=(16,), dtype=np.float32) * image_size)
        loc_mae_pixel = np.array(loc_mae_pixel)
        loc_mae_pixel_x = loc_mae_pixel[:, [0, 2]]
        loc_mae_pixel_y = loc_mae_pixel[:, [1, 3]]
        loc_mae_pixel = np.sqrt(np.square(loc_mae_pixel_x) + np.square(loc_mae_pixel_y))
        bb8_mae_pixel = np.array(bb8_mae_pixel)
        bb8_mae_pixel_x = bb8_mae_pixel[:, [0, 2, 4, 6, 8, 10, 12, 14]]
        bb8_mae_pixel_y = bb8_mae_pixel[:, [1, 3, 5, 7, 9, 11, 13, 15]]
        bb8_mae_pixel = np.sqrt(np.square(bb8_mae_pixel_x) + np.square(bb8_mae_pixel_y))

        valid_count = np.sum(cls_target >= 0)
        box_count = np.sum(cls_target > 0)
        # overall accuracy & object accuracy
        label = cls_target.flatten()
        # in case you have a 'other' class
        label[np.where(label >= cls_prob.shape[1])] = 0
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1])).asnumpy()
        prob = prob[mask, indices]

        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count
        # loc_smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += box_count * 4
        # loc_mae
        self.sum_metric[2] += np.sum(loc_mae)
        self.num_inst[2] += box_count * 4
        # loc_mae_pixel
        self.sum_metric[3] += np.sum(loc_mae_pixel)
        self.num_inst[3] += loc_mae_pixel.size
        # bb8_smoothl1loss
        self.sum_metric[4] += np.sum(bb8_loss)
        self.num_inst[4] += box_count * 16
        # bb8_mae
        self.sum_metric[5] += np.sum(bb8_mae)
        self.num_inst[5] += box_count * 16
        # bb8_mae_pixel
        self.sum_metric[6] += np.sum(bb8_mae_pixel)
        self.num_inst[6] += bb8_mae_pixel.size

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)
