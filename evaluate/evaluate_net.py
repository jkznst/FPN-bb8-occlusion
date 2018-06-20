import os
import pickle
import sys
import importlib
import mxnet as mx
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from dataset.iterator import DetRecordIter
from config.config import cfg
from evaluate.eval_metric import MApMetric, VOC07MApMetric, PoseMetric
import logging
from symbol.symbol_factory import get_symbol
from MultiBoxDetection import BB8MultiBoxDetection


def show_BB8(image, pred_BB8_image_coordinates_list, cids, plot_path):
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    # fig = plt.figure()
    # (3L, 256L, 256L) => (256L, 256L, 3L)
    img = image
    img = cv2.resize(img, dsize=(640, 480))
    # img = np.flip(img, axis=2)
    plt.imshow(img)

    pred_BB8_image_coordinates_list *= np.array([640, 480]).reshape((1,2,1))
    # 蓝色 - 'b'
    # 绿色 - 'g'
    # 红色 - 'r'
    # 青色 - 'c'
    # 品红 - 'm'
    # 黄色 - 'y'
    # 黑色 - 'k'
    # 白色 - 'w'
    color_dict = {0: 'g',   # green -- ape
                  1: 'm',   # magenta -- benchvise
                  2: 'k',   # black -- can
                  3: 'y',   # yellow -- cat
                  4: 'b',   # blue -- driller
                  5: 'r',   # red -- duck
                  6: 'c',   # cyan -- glue
                  7: 'w'}   # white -- holepuncher

    for cid, pred_BB8_image_coordinates in zip(cids, pred_BB8_image_coordinates_list):
        color = color_dict[cid]

        rect6 = plt.Line2D([pred_BB8_image_coordinates[0, 0], pred_BB8_image_coordinates[0, 1],
                            pred_BB8_image_coordinates[0, 2], pred_BB8_image_coordinates[0, 3],
                            pred_BB8_image_coordinates[0, 0]],
                           [pred_BB8_image_coordinates[1, 0], pred_BB8_image_coordinates[1, 1],
                            pred_BB8_image_coordinates[1, 2], pred_BB8_image_coordinates[1, 3],
                            pred_BB8_image_coordinates[1, 0]],
                           linewidth=2, color=color)
        rect7 = plt.Line2D([pred_BB8_image_coordinates[0, 4], pred_BB8_image_coordinates[0, 5],
                            pred_BB8_image_coordinates[0, 6], pred_BB8_image_coordinates[0, 7],
                            pred_BB8_image_coordinates[0, 4]],
                           [pred_BB8_image_coordinates[1, 4], pred_BB8_image_coordinates[1, 5],
                            pred_BB8_image_coordinates[1, 6], pred_BB8_image_coordinates[1, 7],
                            pred_BB8_image_coordinates[1, 4]],
                           linewidth=2, color=color)
        rect8 = plt.Line2D([pred_BB8_image_coordinates[0, 0], pred_BB8_image_coordinates[0, 4]],
                           [pred_BB8_image_coordinates[1, 0], pred_BB8_image_coordinates[1, 4]],
                           linewidth=2, color=color)
        rect9 = plt.Line2D([pred_BB8_image_coordinates[0, 1], pred_BB8_image_coordinates[0, 5]],
                           [pred_BB8_image_coordinates[1, 1], pred_BB8_image_coordinates[1, 5]],
                           linewidth=2, color=color)
        rect10 = plt.Line2D([pred_BB8_image_coordinates[0, 2], pred_BB8_image_coordinates[0, 6]],
                           [pred_BB8_image_coordinates[1, 2], pred_BB8_image_coordinates[1, 6]],
                           linewidth=2, color=color)
        rect11 = plt.Line2D([pred_BB8_image_coordinates[0, 3], pred_BB8_image_coordinates[0, 7]],
                           [pred_BB8_image_coordinates[1, 3], pred_BB8_image_coordinates[1, 7]],
                           linewidth=2, color=color)
        axes.add_line(rect6)
        axes.add_line(rect7)
        axes.add_line(rect8)
        axes.add_line(rect9)
        axes.add_line(rect10)
        axes.add_line(rect11)
    axes.axes.get_xaxis().set_visible(False)
    axes.axes.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig(plot_path)
    plt.close(fig)


def evaluate_net(net, path_imgrec, num_classes, mean_pixels, data_shape,
                 model_prefix, epoch, ctx=mx.cpu(), batch_size=1,
                 path_imglist="", nms_thresh=0.45, force_nms=False,
                 ovp_thresh=0.5, use_difficult=False, class_names=None,
                 voc07_metric=False, frequent=20):
    """
    evalute network given validation record file

    Parameters:
    ----------
    net : str or None
        Network name or use None to load from json without modifying
    path_imgrec : str
        path to the record validation file
    path_imglist : str
        path to the list file to replace labels in record file, optional
    num_classes : int
        number of classes, not including background
    mean_pixels : tuple
        (mean_r, mean_g, mean_b)
    data_shape : tuple or int
        (3, height, width) or height/width
    model_prefix : str
        model prefix of saved checkpoint
    epoch : int
        load model epoch
    ctx : mx.ctx
        mx.gpu() or mx.cpu()
    batch_size : int
        validation batch size
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : boolean
        whether suppress different class objects
    ovp_thresh : float
        AP overlap threshold for true/false postives
    use_difficult : boolean
        whether to use difficult objects in evaluation if applicable
    class_names : comma separated str
        class names in string, must correspond to num_classes if set
    voc07_metric : boolean
        whether to use 11-point evluation as in VOC07 competition
    frequent : int
        frequency to print out validation status
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    #model_prefix += '_' + str(data_shape[1])

    # iterator
    eval_iter = DetRecordIter(path_imgrec, batch_size, data_shape, mean_pixels=mean_pixels,
                              label_pad_width=350, path_imglist=path_imglist, **cfg.valid)
    # model params
    load_net, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
    # network
    if net is None:
        net = load_net
    else:
        net = get_symbol(net, data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_nms)
    if not 'label' in net.list_arguments():
        label = mx.sym.Variable(name='label')
        net = mx.sym.Group([net, label])

    # init module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
        fixed_param_names=net.list_arguments())
    mod.bind(data_shapes=eval_iter.provide_data, label_shapes=eval_iter.provide_label)
    mod.set_params(args, auxs, allow_missing=False, force_init=True)

    # run evaluation
    if voc07_metric:
        metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names,
                                roc_output_path=os.path.join(os.path.dirname(model_prefix), 'roc'))
    else:
        metric = MApMetric(ovp_thresh, use_difficult, class_names,
                            roc_output_path=os.path.join(os.path.dirname(model_prefix), 'roc'))

    posemetric = PoseMetric(LINEMOD_path='/data/ZHANGXIN/DATASETS/SIXD_CHALLENGE/LINEMOD/', classes=class_names)

    # visualize bb8 results
    # for nbatch, eval_batch in tqdm(enumerate(eval_iter)):
    #     mod.forward(eval_batch)
    #     preds = mod.get_outputs(merge_multi_context=True)
    #
    #     labels = eval_batch.label[0].asnumpy()
    #     # get generated multi label from network
    #     cls_prob = preds[0]
    #     loc_pred = preds[4]
    #     bb8_pred = preds[5]
    #     anchors = preds[6]
    #
    #     bb8dets = BB8MultiBoxDetection(cls_prob, loc_pred, bb8_pred, anchors, nms_threshold=0.5, force_suppress=False,
    #                                   variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
    #     bb8dets = bb8dets.asnumpy()
    #
    #     for nsample, sampleDet in enumerate(bb8dets):
    #         image = eval_batch.data[0][nsample].asnumpy()
    #         image += np.array(mean_pixels).reshape((3, 1, 1))
    #         image = np.transpose(image, axes=(1, 2, 0))
    #         draw_dets = []
    #         draw_cids = []
    #
    #         for instanceDet in sampleDet:
    #             if instanceDet[0] == -1:
    #                 continue
    #             else:
    #                 cid = instanceDet[0].astype(np.int16)
    #                 indices = np.where(sampleDet[:, 0] == cid)[0]
    #
    #                 if indices.size > 0:
    #                     draw_dets.append(sampleDet[indices[0], 6:])
    #                     draw_cids.append(cid)
    #                     sampleDet = np.delete(sampleDet, indices, axis=0)
    #                     show_BB8(image / 255., np.transpose(draw_dets[-1].reshape((-1, 8, 2)), axes=(0,2,1)), [cid],
    #                              plot_path='./output/bb8results/{:04d}_{}'.format(nbatch * batch_size + nsample, class_names[cid]))
    #
    #         # draw_dets = np.array(draw_dets)
    #         # draw_cids = np.array(draw_cids)
    #
    #         # show_BB8(image / 255., np.transpose(draw_dets.reshape((-1, 8, 2)), axes=(0,2,1)), draw_cids,
    #         #          plot_path='./output/bb8results/{:04d}'.format(nbatch * batch_size + nsample))

    # quantitive results
    results = mod.score(eval_iter, [metric, posemetric], num_batch=None,
                        batch_end_callback=mx.callback.Speedometer(batch_size,
                                                                   frequent=frequent,
                                                                   auto_reset=False))

    results_save_path = os.path.join(os.path.dirname(model_prefix), 'evaluate_results')
    with open(results_save_path, 'w') as f:
        for k, v in results:
            print("{}: {}".format(k, v))
            f.write("{}: {}\n".format(k, v))
        f.close()

    reproj_save_path = os.path.join(os.path.dirname(model_prefix), 'reprojection_error')
    with open(reproj_save_path, 'wb') as f:
        # for k, v in metric.Reproj.items():
        #     f.write("{}: {}\n".format(k, v))
        pickle.dump(posemetric.Reproj, f, protocol=2)
        f.close()

    count_save_path = os.path.join(os.path.dirname(model_prefix), 'gt_count')
    with open(count_save_path, 'wb') as f:
        # for k, v in metric.counts.items():
        #     f.write("{}: {}\n".format(k, v))
        pickle.dump(posemetric.counts, f, protocol=2)
        f.close()