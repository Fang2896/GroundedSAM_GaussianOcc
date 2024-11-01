# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import pickle
import pdb
import cv2
import torch

from .mono_dataset import MonoDataset


class DDADDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(DDADDataset, self).__init__(*args, **kwargs)

        self.split = 'train' if self.is_train else 'val'
        self.dataroot = self.opt.dataroot

        self.rgb_path = os.path.join(self.opt.dataroot, 'raw_data')
        self.depth_path = os.path.join(self.opt.dataroot, 'depth')
        self.mask_path = os.path.join(self.opt.dataroot, 'mask')
        self.match_path = os.path.join(self.opt.dataroot, 'match')
        self.point_cloud_path = os.path.join(self.opt.dataroot, 'point_cloud2023')
        # self.point_cloud_label_path = '/data/ggeoinfo/Wanshui_BEV/data/ddad/label/point_cloud_{}_label_52_0.0_center_g_fix_num30_new'.format(self.split)
        self.point_cloud_label_path = os.path.join(self.opt.dataroot, 'label/point_cloud_{}_label_52_0.0_center_g_fix_num30_new'.format(self.split))

        if self.opt.data_type == 'mini':
            with open('datasets/ddad/{}_mini.txt'.format(self.split), 'r') as f:
                self.filenames = f.readlines()

        elif self.opt.data_type == 'tiny':
            with open('datasets/ddad/{}_tiny.txt'.format(self.split), 'r') as f:
                self.filenames = f.readlines()

        else:
            with open('datasets/ddad/{}.txt'.format(self.split), 'r') as f:
                self.filenames = f.readlines()

        with open('datasets/ddad/info_{}.pkl'.format(self.split), 'rb') as f:

            self.info = pickle.load(f)

            '''
            train 的数据过滤过，确保了同时拥有前后帧
            
            '''

            # self.info['1568694100173460']  -> dict_keys(['scene_name', 'context', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09'])
            # self.info['1568694100173460']['context'] -> ['1568694100073459', '1568694100273460']
            # self.info['1568694100373461']['context']
            # pdb.set_trace()

        # with open('datasets/ddad/info_train.pkl', 'rb') as f: info = pickle.load(f)

        self.camera_ids = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right']
        self.camera_names = ['CAMERA_01', 'CAMERA_05', 'CAMERA_07', 'CAMERA_09', 'CAMERA_08', 'CAMERA_06']

        # 1568694100873460 : Quaternion(0.5101537704467773, -0.5161614418029785, 0.48277202248573303, -0.4901547133922577), 'tvec': array([1.6692396 , 0.27147174, 1.4880265 ]
        # 1568694100773459 : Quaternion(0.5101537704467773, -0.5161614418029785, 0.48277202248573303, -0.4901547133922577), 'tvec': array([1.6692396 , 0.27147174, 1.4880265 ]
        # 相邻帧之间的相机外参假设不变， 收集数据的时候似乎是每几周重新矫正一次相机外参
    
    def get_info(self, inputs, index_temporal, do_flip):
        inputs[("color", 0, -1)] = []
        inputs[("pose_spatial")] = []
        inputs["color_image_filepath"] = []

        if self.is_train:
            if self.opt.use_sfm_spatial:
                inputs["match_spatial"] = []


            for idx, i in enumerate(self.frame_idxs[0:]):
                inputs[("color", i, -1)] = []
                inputs[("pose_spatial", i)] = []

                # if self.opt.gt_pose:
                #     inputs[("pose_temporal", i)] = []

            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = [] 
            
            inputs['mask_ori'] = []
            inputs['depth'] = []
            inputs['point_cloud'] = []
            inputs['point_cloud_path'] = []

        else:
            inputs[('K_ori', 0)] = [] 
            inputs['depth'] = []
            inputs['mask_ori'] = []

            if self.opt.evl_score:
                inputs['point_cloud'] = []
                inputs['point_cloud_path'] = []
                inputs['point_cloud_label'] = []

            if self.opt.use_t != 'No':
                for idx, i in enumerate(self.frame_idxs_t[0:]):
                    inputs[("color", i, -1)] = []


        inputs['width_ori'], inputs['height_ori'], inputs['id'] = [], [], []

        scene_id = self.info[index_temporal]['scene_name']
        inputs["scene_id"] = scene_id

        # if self.opt.use_t != 'No':
        #     # for idx, i in enumerate(self.frame_idxs_t[0:2]):
        #     for idx, i in enumerate(self.frame_idxs[1:]):
        #         inputs[('gt_pose', i)] = []


        # if self.opt.use_t != 'No' or self.opt.gt_pose:
        #     # for idx, i in enumerate(self.frame_idxs):
        #     gt_pose_0 = self.info[index_temporal]['CAMERA_01']['pose']['quat'].transformation_matrix
        #     gt_pose_0[:3, 3] = self.info[index_temporal]['CAMERA_01']['pose']['tvec']
        #     inputs["gt_pose", 0] = torch.from_numpy(gt_pose_0.astype(np.float32)).unsqueeze(0)

            # index_temporal_i = self.info[index_temporal]['context'][0]  # idx: 0 (前一帧),1 (后一帧)
            #
            # if index_temporal_i == -1: # 第一帧 pose 不变
            #     inputs["gt_pose", -1] = inputs["gt_pose", 0]
            # else:
            #     gt_pose_0 = self.info[index_temporal_i]['CAMERA_01']['pose']['quat'].transformation_matrix
            #     gt_pose_0[:3, 3] = self.info[index_temporal_i]['CAMERA_01']['pose']['tvec']
            #     inputs["gt_pose", -1] = torch.from_numpy(gt_pose_0.astype(np.float32)).unsqueeze(0)
            #     # 需要两帧之间计算每个相机的相对位姿


            # for idx, i in enumerate(self.frame_idxs[1:]):

            #     index_temporal_i = self.info[index_temporal]['context'][idx]  # idx: 0 (前一帧),1 (后一帧)

            #     if index_temporal_i == -1: # test 的时候 第一帧 pose 不变
            #         inputs["gt_pose", i] = inputs["gt_pose", 0]
            #     else:
            #         gt_pose_i = self.info[index_temporal_i]['CAMERA_01']['pose']['quat'].transformation_matrix
            #         gt_pose_i[:3, 3] = self.info[index_temporal_i]['CAMERA_01']['pose']['tvec']
            #         inputs["gt_pose", i] = torch.from_numpy(gt_pose_i.astype(np.float32)).unsqueeze(0)
            #         # 需要两帧之间计算每个相机的相对位姿


        # # for train label
        # if self.is_train:
        #     point_cloud = np.load(os.path.join(self.point_cloud_path, scene_id, 'LIDAR', index_temporal + '.npy'))
        #     inputs['point_cloud'].append(point_cloud)
        #     inputs['point_cloud_path'].append(str(os.path.join(self.point_cloud_path, scene_id, 'LIDAR', index_temporal + '.npy')))


        # if self.opt.evl_score and not self.is_train:

        #     point_cloud = np.load(os.path.join(self.point_cloud_path, scene_id, 'LIDAR', index_temporal + '.npy'))
        #     inputs['point_cloud'].append(point_cloud)
        #     inputs['point_cloud_path'].append(str(os.path.join(self.point_cloud_path, scene_id, 'LIDAR', index_temporal + '.npy')))

        #     # label
        #     point_cloud_label = np.load(os.path.join(self.point_cloud_label_path, scene_id, 'LIDAR', index_temporal + '.npy'),  allow_pickle=True)
        #     point_cloud_label = dict(point_cloud_label.item())
        #     inputs['point_cloud_label'].append(point_cloud_label)


        for index_spatial in range(6):
            # inputs['id'].append(self.camera_ids[index_spatial])
            # color = self.loader(os.path.join(self.rgb_path, scene_id, 'rgb', 
            #                     self.camera_names[index_spatial], index_temporal + '.png'))
            # inputs['width_ori'].append(color.size[0])
            # inputs['height_ori'].append(color.size[1])
            
        
            # if not self.is_train:
            #     depth = np.load(os.path.join(self.depth_path, scene_id, 'depth',
            #                 self.camera_names[index_spatial], index_temporal + '.npy'))
            #     inputs['depth'].append(depth.astype(np.float32))

            # else:

            #     depth = np.load(os.path.join(self.depth_path + '_train_new', scene_id, 'depth',
            #                                  self.camera_names[index_spatial], index_temporal + '.npy'))
            #     inputs['depth'].append(depth.astype(np.float32))


            # if do_flip:
            #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
            # inputs[("color", 0, -1)].append(color)

            # # if self.is_train or self.opt.volume_depth:

            # pose_0_spatial = self.info[index_temporal][self.camera_names[index_spatial]]['extrinsics']['quat'].transformation_matrix
            # pose_0_spatial[:3, 3] = self.info[index_temporal][self.camera_names[index_spatial]]['extrinsics']['tvec']

            # inputs["pose_spatial"].append(pose_0_spatial.astype(np.float32))

    
            # K = np.eye(4).astype(np.float32)
            # K[:3, :3] = self.info[index_temporal][self.camera_names[index_spatial]]['intrinsics']
            # inputs[('K_ori', 0)].append(K)

            # mask = cv2.imread(os.path.join(self.mask_path, self.camera_names[index_spatial], scene_id, 'mask.png'))
            # inputs["mask_ori"].append(mask)

            if self.is_train:
                for idx, i in enumerate(self.frame_idxs[0:1]): # [0, -1, 1]
                    index_temporal_i = self.info[index_temporal]['context'][idx] # idx: 0 (前一帧),1 (后一帧)

                    # K = np.eye(4).astype(np.float32)
                    # K[:3, :3] = self.info[index_temporal_i][self.camera_names[index_spatial]]['intrinsics']
                    # inputs[('K_ori', i)].append(K)

                    color_image_filepath = os.path.join(self.rgb_path, scene_id, 'rgb', 
                                    self.camera_names[index_spatial], index_temporal + '.png')

                    # color = self.loader(color_image_filepath)
                    
                    # if do_flip:
                    #     color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
                    inputs["color_image_filepath"].append(color_image_filepath)
                    # inputs[("color", i, -1)].append(color)
                    # pose_i_spatial = self.info[index_temporal][self.camera_names[(index_spatial+i)%6]]['extrinsics']['quat'].transformation_matrix # 这里用的还是相邻相机： index_temporal
                    # pose_i_spatial[:3, 3] = self.info[index_temporal][self.camera_names[(index_spatial+i)%6]]['extrinsics']['tvec']
                    # gt_pose_spatial = np.linalg.inv(pose_i_spatial) @ pose_0_spatial
                    # inputs[("pose_spatial", i)].append(gt_pose_spatial.astype(np.float32))

  