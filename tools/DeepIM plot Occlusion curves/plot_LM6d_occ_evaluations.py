from __future__ import division
import matplotlib.pyplot as plt
import cPickle
import pickle
import numpy as np
import scipy.io as sio
import os
cur_path = os.path.abspath(os.path.dirname(__file__))

# use_extents = False #########################
# network = 'resnet50m'
iter = 4
font_size = 15
linewidth = 4

# classes = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
classes = ['ape', 'benchvise', 'can', 'cat', 'driller', 'duck', 'glue', 'holepuncher']

# result_dir = os.path.join(cur_path, '../data/LINEMOD_6D/LM6d_occ_results')

# if use_extents:
#     ours_path = os.path.join(result_dir, 'arp_2d_xys_ours_extents.pkl')
#     init_pose_path = os.path.join(result_dir, 'arp_2d_xys_init_pose_extents.pkl')
#     after_ICP_path = os.path.join(result_dir, 'arp_2d_xys_after_ICP_extents.pkl')
# else:
after_DeepIM_path = os.path.join(cur_path, 'arp_2d_xys_DeepIM.pkl')
init_pose_path = os.path.join(cur_path, 'arp_2d_xys_init_pose.pkl')
after_ICP_path = os.path.join(cur_path, 'arp_2d_xys_after_ICP.pkl')
ours_reprojectionerror_resnet50m_300_path = os.path.join(cur_path, 'reprojection_error_ours_resnet50m_300.pkl')
ours_reprojectionerror_resnet50m_512_path = os.path.join(cur_path, 'reprojection_error_ours_resnet50m_512.pkl')
ours_reprojectionerror_vgg16_reduced_300_path = os.path.join(cur_path, 'reprojection_error_ours_vgg16_reduced_300.pkl')
ours_reprojectionerror_vgg16_reduced_512_path = os.path.join(cur_path, 'reprojection_error_ours_vgg16_reduced_512.pkl')
ours_gtcounts_path = os.path.join(cur_path, 'gt_counts_ours.pkl')


with open(after_DeepIM_path, 'rb') as f:
    xys_DeepIM = cPickle.load(f)
with open(init_pose_path, 'rb') as f:
    xys_init_pose = cPickle.load(f)
with open(after_ICP_path, 'rb') as f:
    xys_after_ICP = cPickle.load(f)
with open(ours_reprojectionerror_resnet50m_300_path, 'rb') as f:
    ours_reprojectionerror_resnet50m_300 = pickle.load(f)
with open(ours_reprojectionerror_resnet50m_512_path, 'rb') as f:
    ours_reprojectionerror_resnet50m_512 = pickle.load(f)
with open(ours_reprojectionerror_vgg16_reduced_300_path, 'rb') as f:
    ours_reprojectionerror_vgg16_reduced_300 = pickle.load(f)
with open(ours_reprojectionerror_vgg16_reduced_512_path, 'rb') as f:
    ours_reprojectionerror_vgg16_reduced_512 = pickle.load(f)
with open(ours_gtcounts_path, 'rb') as f:
    ours_gtcounts = pickle.load(f)



## BB8 results
# yxs = sio.loadmat(os.path.join(cur_path, 'bb8_yxs.mat'))
# yxs = yxs['res']
# N = yxs.shape[0]
# yxs_bb8 = {}
# for i in range(N):
#     cls_name = str(yxs[i,0][0])
#     yxs_i = yxs[i, 1]
#     # plt.plot(yxs_i[:, 1], yxs_i[:, 0])
#     # plt.hold(True)
#     yxs_bb8[cls_name] = yxs_i

# print(yxs_bb8['holepuncher'])


# seamless results
def get_seamless_res_old():
    threshs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    ape = [0.00, 0.00, 0.00, 0.00, 0.41, 3.47, 18.16, 51.63, 83.88, 98.78, 99.80, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00,
           100.00, 100.00, 100.00, 100.00]
    can = [0.00, 0.00, 0.00, 0.00, 0.00, 0.13, 0.63, 1.26, 2.40, 4.55, 11.25, 20.23, 33.50, 52.59, 69.53, 82.93, 90.77, 95.70,
           98.48, 99.87, 100.00]
    cat = [0.00, 0.00, 0.00, 8.03, 42.75, 77.98, 92.23, 97.41, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00,
           100.00, 100.00, 100.00, 100.00]
    driller = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.12, 0.35, 1.74, 4.87, 6.84, 12.41, 21.00, 31.67, 46.06, 61.02, 70.88,
               81.90, 87.35]
    duck = [0.00, 0.18, 5.64, 20.18, 39.82, 68.36, 91.64, 96.73, 99.27, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00,
            100.00, 100.00, 100.00, 100.00, 100.00]
    glue = [0.00, 0.00, 0.00, 0.00, 0.00, 1.64, 2.46, 7.38, 23.77, 43.44, 59.84, 71.31, 81.15, 91.80, 93.44, 98.36, 99.18,
            100.00, 100.00, 100.00, 100.00]
    holepuncher = [0.00, 0.00, 0.89, 4.44, 13.68, 28.42, 43.34, 61.63, 74.42, 84.37, 92.18, 96.27, 98.58, 99.47, 100.00, 100.00,
            100.00, 100.00, 100.00, 100.00, 100.00]

    xys_seamless = {'x': np.array(threshs),
                    'ape': np.array(ape),
                    'can': np.array(can),
                    'cat': np.array(cat),
                    'driller': np.array(driller),
                    'duck': np.array(duck),
                    'glue': np.array(glue),
                    'holepuncher': np.array(holepuncher)}
    return xys_seamless

def get_seamless_res():
    threshs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ape = [0.00, 7.01, 40.43, 59.91, 68.46, 72.14, 73.85, 74.79, 75.38, 75.90, 76.67]
    can = [0.00, 11.18, 57.83, 79.95, 85.75, 88.65, 90.31, 91.71, 93.12, 93.54, 93.70]
    cat = [0.00, 3.62, 23.25, 39.34, 48.61, 53.50, 56.36, 57.96, 59.14, 60.49, 61.33]
    driller = [0.00, 1.40, 17.38, 39.87, 62.93, 80.48, 89.62, 94.56, 95.55, 96.21, 96.54]
    duck = [0.00, 5.07, 18.20, 30.88, 55.03, 75.15, 81.36, 83.20, 83.73, 83.99, 84.25]
    glue = [0.00, 6.53, 26.91, 39.65, 46.51, 50.06, 51.94, 53.27, 53.49, 54.15, 54.60]
    holepuncher = [0.00, 8.26, 39.50, 53.31, 62.56, 68.02, 74.63, 80.66, 85.54, 89.42, 91.32]


    xys_seamless = {'x': np.array(threshs),
                    'ape': np.array(ape),
                    'can': np.array(can),
                    'cat': np.array(cat),
                    'driller': np.array(driller),
                    'duck': np.array(duck),
                    'glue': np.array(glue),
                    'holepuncher': np.array(holepuncher)}
    return xys_seamless

def get_SSD_BB8_res(reprojection_error, gt_counts):
    threshs = np.arange(0., 50., 0.1)

    name = ['ape', 'benchvise', 'can', 'cat', 'driller', 'duck', 'glue', 'holepuncher']

    xys_SSD_BB8 = {'x': np.array(threshs),
                   'ape': np.zeros_like(threshs),
                   'benchvise': np.zeros_like(threshs),
                   'can': np.zeros_like(threshs),
                   'cat': np.zeros_like(threshs),
                   'driller': np.zeros_like(threshs),
                   'duck': np.zeros_like(threshs),
                   'glue': np.zeros_like(threshs),
                   'holepuncher': np.zeros_like(threshs)}

    for k, errors in reprojection_error.items():
        for error in errors:
            count = np.where(threshs >= error, 1., 0.)
            xys_SSD_BB8[name[k]] += count
        xys_SSD_BB8[name[k]] = xys_SSD_BB8[name[k]] / gt_counts[k] * 100.

    return xys_SSD_BB8


xys_seamless = get_seamless_res()
xys_SSD_BB8_resnet50m_300 = get_SSD_BB8_res(ours_reprojectionerror_resnet50m_300, ours_gtcounts)
xys_SSD_BB8_resnet50m_512 = get_SSD_BB8_res(ours_reprojectionerror_resnet50m_512, ours_gtcounts)
xys_SSD_BB8_vgg16_reduced_300 = get_SSD_BB8_res(ours_reprojectionerror_vgg16_reduced_300, ours_gtcounts)
xys_SSD_BB8_vgg16_reduced_512 = get_SSD_BB8_res(ours_reprojectionerror_vgg16_reduced_512, ours_gtcounts)

for cls_name in classes:
    fig = plt.figure(figsize=(8, 6), dpi=100)

    # if cls_name != 'ape':
    #     continue
    # ------------------------------------------------


    # if not cls_name in yxs_bb8.keys():
    #     yxs_bb8[cls_name] = np.array([[0, 0]])
    #
    # bb8, = plt.plot(yxs_bb8[cls_name][:, 1], yxs_bb8[cls_name][:, 0], label='BB8 w/ gt bbox', linewidth=linewidth)
    if not cls_name in xys_init_pose.keys():
        init_pose, = plt.plot(0, 0, label='PoseCNN', linewidth=linewidth)
    else:
        init_pose, = plt.plot(xys_init_pose[cls_name][0][0], xys_init_pose[cls_name][0][1], label='PoseCNN',
                              linewidth=linewidth)
    # if not cls_name in xys_after_ICP.keys():
    #     after_ICP, = plt.plot(0, 0, label='PoseCNN+ICP', linewidth=linewidth)
    # else:
    #     after_ICP, = plt.plot(xys_after_ICP[cls_name][0][0], xys_after_ICP[cls_name][0][1], label='PoseCNN+ICP',
    #                           linewidth=linewidth)
    if not cls_name in xys_DeepIM.keys():
        after_DeepIM, = plt.plot(0, 0,label='PoseCNN+DeepIM',linewidth=linewidth)
    else:
        after_DeepIM, = plt.plot(xys_DeepIM[cls_name][iter - 1][0], xys_DeepIM[cls_name][iter - 1][1], label='PoseCNN+DeepIM',
                         linewidth=linewidth)

    if not cls_name in xys_seamless.keys():
        seamless, = plt.plot(0, 0, label='Tekin et al.', linewidth=linewidth)
    else:
        seamless,  = plt.plot(xys_seamless['x'], xys_seamless[cls_name], label='Tekin et al.',
                              linewidth=linewidth)

    if not cls_name in xys_SSD_BB8_resnet50m_300.keys():
        SSD_BB8_resnet50m_300, = plt.plot(0, 0, label='OURS-ResNet50m-300', linewidth=linewidth)
    else:
        SSD_BB8_resnet50m_300, = plt.plot(xys_SSD_BB8_resnet50m_300['x'], xys_SSD_BB8_resnet50m_300[cls_name], label='OURS-ResNet50m-300', linewidth=linewidth)

    if not cls_name in xys_SSD_BB8_resnet50m_512.keys():
        SSD_BB8_resnet50m_512, = plt.plot(0, 0, label='OURS-ResNet50m-512', linewidth=linewidth)
    else:
        SSD_BB8_resnet50m_512, = plt.plot(xys_SSD_BB8_resnet50m_512['x'], xys_SSD_BB8_resnet50m_512[cls_name], label='OURS-ResNet50m-512', linewidth=linewidth)

    if not cls_name in xys_SSD_BB8_vgg16_reduced_300.keys():
        SSD_BB8_vgg16_reduced_300, = plt.plot(0, 0, label='OURS-VGG16-300', linewidth=linewidth)
    else:
        SSD_BB8_vgg16_reduced_300, = plt.plot(xys_SSD_BB8_vgg16_reduced_300['x'], xys_SSD_BB8_vgg16_reduced_300[cls_name],
                            label='OURS-VGG16-300', linewidth=linewidth)

    if not cls_name in xys_SSD_BB8_vgg16_reduced_512.keys():
        SSD_BB8_vgg16_reduced_512, = plt.plot(0, 0, label='OURS-VGG16-512', linewidth=linewidth)
    else:
        SSD_BB8_vgg16_reduced_512, = plt.plot(xys_SSD_BB8_vgg16_reduced_512['x'], xys_SSD_BB8_vgg16_reduced_512[cls_name],
                            label='OURS-VGG16-512', linewidth=linewidth)

    plt.xlim(0, 25)
    plt.ylim(0, 100)
    # if cls_name in yxs_bb8.keys(): # eggbox not in BB8
    # plt.legend([ init_pose, after_ICP, seamless, after_DeepIM, SSD_BB8_resnet50m_300, SSD_BB8_vgg16_reduced_300],
    #            [ 'PoseCNN', 'PoseCNN+ICP', 'Tekin et al.', 'PoseCNN+DeepIM', 'OURS-ResNet50m-300', 'OURS-VGG16-300'],
    #            loc='lower right', fontsize=font_size)
    plt.legend([init_pose, seamless, after_DeepIM, SSD_BB8_resnet50m_300, SSD_BB8_resnet50m_512, SSD_BB8_vgg16_reduced_300, SSD_BB8_vgg16_reduced_512],
               ['PoseCNN', 'Tekin et al.', 'PoseCNN+DeepIM', 'OURS-ResNet50m-300', 'OURS-ResNet50m-512', 'OURS-VGG16-300', 'OURS-VGG16-512'],
               loc='lower right', fontsize=font_size)
    # else:
    #     plt.legend([init_pose, ours], ['Initial Pose', 'Ours'], loc='best')
    # fig.suptitle('{}, average re-projection 2d'.format(cls_name))
    plt.xlabel('pixel threshold', fontsize=font_size)
    # plt.ylabel('correctly estimated poses in %')
    plt.ylabel('accuracy', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title(cls_name, fontsize=font_size, fontweight='bold')
    plt.tight_layout()
    # plt.grid(True)
    # plt.show()
    # if use_extents:
    #     plt.savefig(os.path.join(result_dir, '{}_arp2d_extents.png'.format(cls_name)), dpi=fig.dpi)
    # else:
    plt.savefig(os.path.join(cur_path, 'reproj_occlusion_{}.pdf'.format(cls_name)), dpi=fig.dpi)

