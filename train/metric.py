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
        def iou(x, ys):
            """
            Calculate intersection-over-union overlap
            Params:
            ----------
            x : numpy.array
                single box [xmin, ymin ,xmax, ymax]
            ys : numpy.array
                multiple box [[xmin, ymin, xmax, ymax], [...], ]
            Returns:
            -----------
            numpy.array
                [iou1, iou2, ...], size == ys.shape[0]
            """
            ixmin = np.maximum(ys[:, 0], x[0])
            iymin = np.maximum(ys[:, 1], x[1])
            ixmax = np.minimum(ys[:, 2], x[2])
            iymax = np.minimum(ys[:, 3], x[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = (x[2] - x[0]) * (x[3] - x[1]) + (ys[:, 2] - ys[:, 0]) * \
                (ys[:, 3] - ys[:, 1]) - inters
            ious = inters / uni
            ious[uni < 1e-12] = 0  # in case bad boxes
            return ious

        labels = labels[0].asnumpy()
        # get generated multi label from network
        cls_prob = preds[0]
        loc_loss = preds[1].asnumpy()     # smoothL1 loss
        loc_loss_in_use = loc_loss[loc_loss.nonzero()]
        cls_label = preds[2].asnumpy()
        bb8_loss = preds[3].asnumpy()
        loc_pred = preds[4]
        bb8_pred = preds[5]
        anchors = preds[6]
        # anchor_in_use = anchors[anchors.nonzero()]
        bb8dets = BB8MultiBoxDetection(cls_prob, loc_pred, bb8_pred, anchors, nms_threshold=0.5, force_suppress=False,
                                      variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
        bb8dets = bb8dets.asnumpy()

        loc_label = preds[7].asnumpy()
        loc_label_in_use = loc_label[loc_label.nonzero()]
        loc_pred_masked = preds[8].asnumpy()
        loc_pred_in_use = loc_pred_masked[loc_pred_masked.nonzero()]
        loc_mae = preds[9].asnumpy()
        loc_mae_in_use = loc_mae[loc_mae.nonzero()]
        # loc_mae_pixel = np.abs((bb8dets[:, 0, 2:6] - labels[:, 0, 1:5]) * 300)   # need to be refined
        # for each class, only consider the most confident instance
        loc_mae_pixel = []
        bb8_mae_pixel = []
        for sampleDet, sampleLabel in zip(bb8dets, labels):
            for instanceLabel in sampleLabel:
                if instanceLabel[0] < 0:
                    continue
                else:
                    cid = instanceLabel[0]
                    indices = np.where(sampleDet[:, 0] == cid)[0]
                    if indices.size > 0:
                        instanceDet = sampleDet[indices[0]] # only consider the most confident instance
                        loc_mae_pixel.append(np.abs((instanceDet[2:6] - instanceLabel[1:5]) * 300))
                        bb8_mae_pixel.append(np.abs((instanceDet[6:22] - instanceLabel[8:24]) * 300))
        loc_mae_pixel = np.array(loc_mae_pixel)
        bb8_mae_pixel = np.array(bb8_mae_pixel)
        bb8_mae_pixel_x = bb8_mae_pixel[:, [0, 2, 4, 6, 8, 10, 12, 14]]
        bb8_mae_pixel_y = bb8_mae_pixel[:, [1, 3, 5, 7, 9, 11, 13, 15]]
        bb8_mae_pixel = np.sqrt(np.square(bb8_mae_pixel_x) + np.square(bb8_mae_pixel_y))

        bb8_label = preds[10].asnumpy()
        bb8_label_in_use = bb8_label[bb8_label.nonzero()]
        bb8_pred = preds[11].asnumpy()
        bb8_pred_in_use = bb8_pred[bb8_pred.nonzero()]
        bb8_mae = preds[12].asnumpy()
        bb8_mae_in_use = bb8_mae[bb8_mae.nonzero()]
        # bb8_mae_pixel = np.abs((labels[:, 0, 8:24] - bb8dets[:, 0, 6:22]) * 300)  # need to be refined

        # loc_mae_pixel = []
        # bb8_mae_pixel = []
        # # independant execution for each image
        # for i in range(labels.shape[0]):
        #     # get as numpy arrays
        #     label = labels[i]
        #     pred = bb8dets[i]
        #     loc_mae_pixel_per_image = []
        #     bb8_mae_pixel_per_image = []
        #     # calculate for each class
        #     while (pred.shape[0] > 0):
        #         cid = int(pred[0, 0])
        #         indices = np.where(pred[:, 0].astype(int) == cid)[0]
        #         if cid < 0:
        #             pred = np.delete(pred, indices, axis=0)
        #             continue
        #         dets = pred[indices]
        #         pred = np.delete(pred, indices, axis=0)
        #
        #         # ground-truths
        #         label_indices = np.where(label[:, 0].astype(int) == cid)[0]
        #         gts = label[label_indices, :]
        #         label = np.delete(label, label_indices, axis=0)
        #         if gts.size > 0:
        #             found = [False] * gts.shape[0]
        #             for j in range(dets.shape[0]):
        #                 # compute overlaps
        #                 ious = iou(dets[j, 2:6], gts[:, 1:5])
        #                 ovargmax = np.argmax(ious)
        #                 ovmax = ious[ovargmax]
        #                 if ovmax > self.ovp_thresh:
        #                     if not found[ovargmax]:
        #                         loc_mae_pixel_per_image.append(np.abs((dets[j, 2:6] - gts[ovargmax, 1:5]) * 300))   # tp
        #                         bb8_mae_pixel_per_image.append(np.abs((dets[j, 6:22] - gts[ovargmax, 8:24]) * 300))
        #                         found[ovargmax] = True
        #                     else:
        #                         # duplicate
        #                         pass  # fp
        #
        #     loc_mae_pixel.append(np.mean(loc_mae_pixel_per_image, axis=1))
        #     bb8_mae_pixel.append(np.mean(bb8_mae_pixel_per_image, axis=1))


        valid_count = np.sum(cls_label >= 0)
        box_count = np.sum(cls_label > 0)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
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
