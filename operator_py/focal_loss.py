# --------------------------------------------------------
# Focal loss
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by unsky https://github.com/unsky/
# --------------------------------------------------------

"""
Focal loss 
"""

import mxnet as mx
import numpy as np
from mxnet import autograd


class FocalLossOperator(mx.operator.CustomOp):
    def __init__(self,  gamma, alpha):
        super(FocalLossOperator, self).__init__()
        self._gamma = gamma
        self._alpha = alpha 

    def forward(self, is_train, req, in_data, out_data, aux):
      
        cls_preds = in_data[0]      # batchsize x num_class x 8732
        cls_target = in_data[1]     # batchsize x 8732
        self._labels = cls_target

        # pro_ = mx.nd.exp(cls_preds - cls_preds.max(axis=1, keepdims=True))
        # pro_ /= mx.nd.sum(data=pro_, axis=1, keepdims=True)

        prob = mx.nd.softmax(data=cls_preds, axis=1)    # batchsize x num_class x 8732
        self.pro_ = prob
       
        self._pt = mx.nd.pick(data=prob, index=cls_target, axis=1, keepdims=True)   # batchsize x 1 x 8732
 
        ### note!!!!!!!!!!!!!!!!
        # focal loss value is not used in this place we should forward the cls_pro in this layer, 
        # the focal vale should be calculated in metric.py
        # the method is in readme
        #  focal loss (batch_size,num_class)
        #loss_ = -1 * np.power(1 - pro_, self._gamma) * np.log(pro_)
        self.assign(out_data[0],req[0],mx.nd.array(prob))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # method 1
        cls_target = self._labels
        prob = self.pro_
        #i!=j
        pt = self._pt + 1e-14   # batchsize x 1 x 8732
        dx =  self._alpha * mx.nd.power(1 - pt, self._gamma - 1) * (self._gamma * (-1 * pt * prob) * mx.nd.log(pt) + prob * (1 - pt))

        ####i==j
        #reload pt
        pt = self._pt + 1e-14
        # dx[np.repeat(np.arange(prob.shape[0]), repeats=prob.shape[2]), mx.nd.reshape(cls_target, shape=(1, -1)),
        #    np.tile(np.arange(prob.shape[2]), reps=prob.shape[0])] = \
        #     (self._alpha * mx.nd.power(1 - pt, self._gamma) * (self._gamma * pt * mx.nd.log(pt) + pt -1) * (1.0)).reshape(shape=(1,-1))
        for nbatch in range(dx.shape[0]):
            dx_batch = dx[nbatch]
            dx_batch[cls_target[nbatch], np.arange(dx.shape[2])] = \
                (self._alpha * mx.nd.power(1 - pt[nbatch,0], self._gamma) * (self._gamma * pt[nbatch,0] * mx.nd.log(pt[nbatch,0]) + pt[nbatch,0] - 1))
        dx /= (cls_target.shape[0] * cls_target.shape[1] * 2) ##batch
        self.assign(in_grad[0], req[0], dx)
        self.assign(in_grad[1],req[1],0)

        # method 2
        # cls_preds = in_data[0]
        # cls_target = in_data[1]
        #
        # cls_preds.attach_grad()
        # with autograd.record():
        #     prob = mx.nd.softmax(data=cls_preds, axis=1)
        #     pj = prob.pick(cls_target, axis=1, keepdims=True)
        #     focalloss = - 0.25 * ((1 - pj) ** 2.) * pj.log()
        # focalloss.backward()
        # grad = cls_preds.grad
        #
        # self.assign(in_grad[0], req[0], grad)
        # self.assign(in_grad[1], req[1], 0)
 
         

@mx.operator.register('FocalLoss')
class FocalLossProp(mx.operator.CustomOpProp):
    def __init__(self, gamma,alpha):
        super(FocalLossProp, self).__init__(need_top_grad=False)

        self._gamma = float(gamma)
        self._alpha = float(alpha)

    def list_arguments(self):
        return ['data', 'labels']

    def list_outputs(self):
        return ['focal_loss']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        labels_shape = in_shape[1]
        out_shape = data_shape
        return  [data_shape, labels_shape],[out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FocalLossOperator(self._gamma,self._alpha)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
