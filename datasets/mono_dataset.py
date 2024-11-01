# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import pdb

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import time
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms
# import pdb
import open3d as o3d


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
    """
    def __init__(self,
                 opt,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False, volume_depth= False):
        super(MonoDataset, self).__init__()

        self.opt = opt
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.frame_idxs_permant = frame_idxs

        self.is_train = is_train
        self.volume_depth = volume_depth

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)


    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                if len(k) != 3:
                    continue
                n, im, i = k
                inputs[(n + "_aug", im, -1)] = [] # 这里-1 代表原始尺寸

                for i in range(self.num_scales):
                    inputs[(n, im, i)] = []
                    inputs[(n + "_aug", im, i)] = []
                    #print(n, im, i)
                    for index_spatial in range(6):

                        #try:
                        inputs[(n, im, i)].append(self.resize[i](inputs[(n, im, i - 1)][index_spatial]))

                        # except:
                        #
                        #     print(len(inputs[("color", -1, -1)]))
                        #
                        #     exit()

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                if len(k) != 3:
                    continue
                n, im, i = k
                for index_spatial in range(6):
                    aug = color_aug(f[index_spatial])
                    # try:
                    inputs[(n, im, i)][index_spatial] = self.to_tensor(f[index_spatial])
                    inputs[(n + "_aug", im, i)].append(self.to_tensor(aug))

                    # except:
                    #     print('name:', n, im, i)
                    #     print(f[index_spatial])
                
                inputs[(n, im, i)] = torch.stack(inputs[(n, im, i)], dim=0)
                inputs[(n + "_aug", im, i)] = torch.stack(inputs[(n + "_aug", im, i)], dim=0)

    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """

        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5
        # do_color_aug = False
        # do_flip = self.is_train and (not self.opt.use_sfm_spatial) and (not self.opt.joint_pose) and random.random() > 0.5
        do_flip = False

        frame_index = self.filenames[index].strip().split()[0]

        # if frame_index == '1569505864403383':

        #     pdb.set_trace()

        # print('frame_index:', frame_index)
        # print('len filesname:', len(self.filenames))

        self.frame_idxs = self.frame_idxs_permant

        self.get_info(inputs, frame_index, do_flip)

        # # adjusting intrinsics to match each scale in the pyramid
        # if not self.is_train:
        #     self.frame_idxs = [0]

        # for scale in range(self.num_scales):
        #     for frame_id in self.frame_idxs:
        #         inputs[("K", frame_id, scale)] = []
        #         inputs[("inv_K", frame_id, scale)] = []
        #         inputs[("K_render", frame_id, scale)] = []
        #         inputs[("inv_K_render", frame_id, scale)] = []

    
        # for index_spatial in range(6):
        #     for scale in range(self.num_scales):
        #         for frame_id in self.frame_idxs:
        #             K = inputs[('K_ori', frame_id)][index_spatial].copy()
        
        #             K[0, :] *= (self.width // (2 ** scale)) / inputs['width_ori'][index_spatial]
        #             K[1, :] *= (self.height // (2 ** scale)) / inputs['height_ori'][index_spatial]
        #             inv_K = np.linalg.pinv(K)  # 取逆
        #             inputs[("K", frame_id, scale)].append(torch.from_numpy(K))  #  这里是相对于原图进行一次scale
        #             inputs[("inv_K", frame_id, scale)].append(torch.from_numpy(inv_K))

        #             K_render = inputs[('K_ori', frame_id)][index_spatial].copy()
        #             K_render[0, :] *= (self.opt.render_w // (2 ** scale)) / inputs['width_ori'][index_spatial]
        #             K_render[1, :] *= (self.opt.render_h // (2 ** scale)) / inputs['height_ori'][index_spatial]
        #             inputs[("K_render", frame_id, scale)].append(torch.from_numpy(K_render))
        #             inv_K_render = np.linalg.pinv(K_render)  # 取逆

        #             inputs[("inv_K_render", frame_id, scale)].append(torch.from_numpy(inv_K_render))


        # for scale in range(self.num_scales):
        #     for frame_id in self.frame_idxs:
        #         inputs[("K", frame_id, scale)] = torch.stack(inputs[("K", frame_id, scale)], dim=0)
        #         inputs[("inv_K", frame_id, scale)] = torch.stack(inputs[("inv_K", frame_id, scale)], dim=0)
        #         inputs[("K_render", frame_id, scale)] = torch.stack(inputs[("K_render", frame_id, scale)], dim=0)
        #         inputs[("inv_K_render", frame_id, scale)] = torch.stack(inputs[("inv_K_render", frame_id, scale)], dim=0)


        # if do_color_aug:
        #     color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        # else:
        #     color_aug = lambda x: x

        # # self.preprocess(inputs, color_aug)

        # del inputs[("color", 0, -1)]
        # del inputs['width_ori']
        # del inputs['height_ori']

        # if 'depth' in inputs.keys():
        #     inputs['depth'] = torch.from_numpy(inputs['depth'])
        #     # print('depth shape:', inputs['depth'].shape)

        # if 'optical_flow' in inputs.keys():
        #     inputs['optical_flow'] = torch.from_numpy(inputs['optical_flow'])

        #     # print('optical_flow1:', inputs['optical_flow'].shape)

        # if 'pseudo_depth' in inputs.keys():
        #     inputs['pseudo_depth'] = torch.from_numpy(inputs['pseudo_depth'])


        # # if 'sky_mask' in inputs.keys():
        # #     inputs['sky_mask'] = torch.from_numpy(inputs['sky_mask'])

        # if 'pose_spatial' in inputs.keys():
        #     inputs["pose_spatial"] = torch.from_numpy(inputs["pose_spatial"])

        # if self.is_train:

        #     if self.opt.pose_aug != 'No':
        #         inputs["cam2"] = torch.from_numpy(inputs["cam2"])
        #         inputs[('K_ori', 0)] = torch.from_numpy(inputs[('K_ori', 0)])

        # if self.opt.use_fix_mask:

        #     inputs["mask"] = []
            
        #     for i in range(6):
        #         temp = cv2.resize(inputs["mask_ori"][i], (self.width, self.height))
        #         temp = temp[..., 0]
        #         temp = (temp == 0).astype(np.float32)
        #         inputs["mask"].append(temp)
        #     inputs["mask"] = np.stack(inputs["mask"], axis=0)
        #     inputs["mask"] = np.tile(inputs["mask"][:, None], (1, 2, 1, 1))
        #     inputs["mask"] = torch.from_numpy(inputs["mask"])

        #     if do_flip:
        #         inputs["mask"] = torch.flip(inputs["mask"], [3])
        #     del inputs["mask_ori"]


        # if self.volume_depth or (self.opt.o_mask != 'No' and self.opt.gs_depth != 'No') :
        #     with torch.no_grad():
        #         rays_o, rays_d = get_rays_of_a_view(H=self.opt.render_h, W=self.opt.render_w, K=inputs[('K_render', 0, 0)], c2w=inputs['pose_spatial'], ndc=False, inverse_y=True, flip_x=False, flip_y=False, mode='center')
        #         inputs['rays_o', 0] = rays_o
        #         inputs['rays_d', 0] = rays_d

        # if self.opt.use_t != 'No' and self.is_train:
        #     inputs["color_aug", 0, 0] = torch.cat([inputs["color_aug", 0, 0], inputs["color_aug", -1, 0], inputs["color_aug", 1, 0]], dim= 0)

        # # inputs["color_aug", 0, 0] =  inputs["color_aug", -1, 0]

        # inputs["all_cam_center"] = torch.from_numpy(np.array([1.2475059, 0.0673422, 1.5356342])).unsqueeze(0).to(torch.float32) # DDAD


        # if self.opt.evl_score and not self.is_train:
        #     self.get_occupancy_test_label(inputs)


        # if self.opt.surfaceloss and self.is_train:
        #     self.get_occupancy_train_label(inputs)

        # if self.is_train:
        #     del inputs['point_cloud']

        #     for i in self.frame_idxs[1:]:
        #         del inputs[("color", i, -1)]
        #         del inputs[("color_aug", i, -1)]
        #     for i in self.frame_idxs:
        #         del inputs[('K_ori', i)]
        # else:

        #     del inputs[('K_ori', 0)]

        # # print(inputs["color", -1, 0].shape)

        return inputs


    def get_info(self, inputs, index, do_flip):
        raise NotImplementedError


    def get_mask(self, pts_xyz):

        # mask1 = pts_xyz[..., 2] < self.opt.val_reso

        mask1 = pts_xyz[..., 2] < 0.001

        mask2 = pts_xyz[..., 2] >= self.opt.real_size[5]

        xy_range = self.opt.max_depth


        # x
        mask3 = pts_xyz[..., 0] > xy_range
        mask4 = pts_xyz[..., 0] < -xy_range

        # y
        mask5 = pts_xyz[..., 1] > xy_range
        mask6 = pts_xyz[..., 1] < -xy_range

        # mask out the point cloud close to car, especially for nuscenes
        # x
        mask7 = (pts_xyz[..., 0] < 3.5) & (pts_xyz[..., 0] > -0.5)
        # y
        mask8 = (pts_xyz[..., 1] < 1.0) & (pts_xyz[..., 1] > -1.0)

        mask9 = mask7 & mask8

        mask = mask1 + mask2 + mask3 + mask4 + mask5 + mask6 + mask9

        return mask


    def get_occupancy_test_label(self, inputs):

        # if 1 and 'ddad' in self.opt.model_name:
        if 1:
            inputs['all_empty'] = inputs['point_cloud_label'][0]['all_empty']
            inputs['True_center'] = inputs['point_cloud_label'][0]['True_center']

            #if self.opt.center_label:
            # inputs['True_voxel_group'] = inputs['point_cloud_label'][0]['True_voxel_group']

            # for depth metric
            inputs['val_depth_empty'] = inputs['point_cloud_label'][0]['val_depth_empty']
            inputs['mask'] = inputs['point_cloud_label'][0]['mask']
            inputs['total_depth_empty'] = inputs['point_cloud_label'][0]['total_depth_empty']
            inputs['surface_point'] = inputs['point_cloud_label'][0]['surface_point']
            inputs['origin'] = inputs['point_cloud_label'][0]['origin']

            return None


        all_pose, pts_xyz = inputs['pose_spatial'], inputs['point_cloud'][0]
        np.random.seed(0)

        # voxel_size = self.opt.max_depth/self.opt.voxels_size[0]  # 乘于2，再除于2，相消
        voxel_size = self.opt.val_reso
        stride = voxel_size  # empty stride
        mask = self.get_mask(pts_xyz)

        GT_point = pts_xyz[~mask]
        pts_xyz = GT_point

        # point cloud down sample
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_xyz)
        pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)
        pcd_down_sample_point = np.asarray(pcd.points)

        # get all camera center
        all_cam_center = np.array([all_pose[:, 0, 3].mean(), all_pose[:, 1, 3].mean(), all_pose[:, 2, 3].mean()])

        if self.opt.val_binary:
            # for classification label

            # total_vertex = []
            # total_center = []

            # http://www.open3d.org/docs/0.8.0/python_api/open3d.geometry.VoxelGrid.html
            # 进行判断的点先进行降采样，其中降采样为voxel分辨率的1/4
            total_center = np.array(pcd_down_sample_point)
            pcd_down_sample_point = total_center
            inputs['True_center'] = torch.from_numpy(total_center)

            # for group
            # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
            # total_voxel_group = []
            # for i in range(pcd_down_sample_point.shape[0]):
            #     # get group point
            #     vertex = np.asarray(voxel_grid.get_voxel_bounding_points(voxel_grid.get_voxel(pcd_down_sample_point[i, :])))
            #     # get voxel center
            #     center_point = voxel_grid.get_voxel_center_coordinate(voxel_grid.get_voxel(pcd_down_sample_point[i, :]))
            #     instance_group = np.concatenate((vertex, center_point[np.newaxis, :]), axis=0)
            #     # 需要注意有无量化的误差，导致误判
            #     total_voxel_group.append(instance_group)
            #
            # total_voxel_group = np.array(total_voxel_group)
            # total_voxel_group = np.unique(total_voxel_group, axis=0)
            # inputs['True_voxel_group'] = torch.from_numpy(total_voxel_group)
            #

            # 如果调整成距离表达则无这个影响
            if inputs['True_center'].shape[0] < 20:
                inputs['val_depth_empty'] = 1
                inputs['mask'] = 1
                inputs['total_depth_empty'] = 1
                inputs['surface_point'] = 1
                inputs['origin'] = 1
                inputs['all_empty'] = 1
                inputs['True_voxel_group'] = 1

            else:
                # get empty point
                total_empty_point = []
                # sample = 'fix_step'
                sample = 'fix_num'

                if sample == 'fix_step':
                    for i in range(len(pcd_down_sample_point)):
                        vector = pcd_down_sample_point[i] - all_cam_center
                        length = np.linalg.norm(vector)
                        norm_vector = vector / length

                        # pdb.set_trace() 这里可以
                        # for j in range(1, int((length // stride) - (1.5 // stride))):
                        for j in range(1, min(20, int((length // stride) - (1.5 // stride)))):

                            sampled_point = pcd_down_sample_point[i] - stride * j * norm_vector
                            sampled_point = sampled_point.tolist()
                            total_empty_point.append(sampled_point)

                    total_empty_point = np.array(total_empty_point)
                    total_empty_point = torch.from_numpy(total_empty_point)


                elif sample == 'fix_num':

                    # N_step = 30
                    # pts_xyz = torch.from_numpy(pcd_down_sample_point).float()
                    # vector = pts_xyz - all_cam_center
                    # length = vector.norm(dim=1)
                    # norm_vector = vector / length[:, None]
                    #
                    # mean_step = length / N_step
                    # train_rng = torch.arange(N_step)[None].float()
                    #
                    # instance_step = mean_step[:, None] * train_rng.repeat(pts_xyz.shape[0], 1)
                    # pts_xyz_all = pts_xyz[..., None, :] - norm_vector[..., None, :] * instance_step[..., None]
                    # total_empty_point = pts_xyz_all[:, 1:-1, :].flatten(0, 1)  # 搞一个mask 在这里边的就去掉


                    N_step = 30
                    pts_xyz = torch.from_numpy(pcd_down_sample_point).float()
                    vector = pts_xyz - all_cam_center
                    length = vector.norm(dim=1)
                    norm_vector = vector / length[:, None]

                    # get last empty with voxel size step
                    last_empty_point = pts_xyz - stride * norm_vector
                    empty_vector = last_empty_point - all_cam_center
                    empty_length = empty_vector.norm(dim=1)
                    mean_step = empty_length / N_step

                    train_rng = torch.arange(N_step)[None].float()
                    instance_step = mean_step[:, None] * train_rng.repeat(last_empty_point.shape[0], 1)
                    pts_xyz_all = last_empty_point[..., None, :] - norm_vector[..., None, :] * instance_step[..., None]
                    total_empty_point = pts_xyz_all[:, 1:-1, :].flatten(0, 1)  # 搞一个mask 在这里边的就去掉

                # check empty
                all_empty = total_empty_point
                # in_mask = voxel_grid.check_if_included(o3d.utility.Vector3dVector(all_empty))  # 可以将车放置一些点作为voxel
                # out_mask = ~np.array(in_mask)
                # all_empty = all_empty[out_mask]

                # mask out colse to car
                # xy_range_1 = 2.0
                # xy_range_2 = 2.0
                # x
                mask7 = (all_empty[:, 0] < 3.5) & (all_empty[:, 0] > -0.5)  # 提前预设点
                # y
                mask8 = (all_empty[:, 1] < 1.0) & (all_empty[:, 1] > -1.0)
                mask = mask7 & mask8
                # o3d.visualization.draw_geometries([pcd])

                all_empty = all_empty[~mask]
                inputs['all_empty'] = all_empty


        if self.opt.val_depth:
            # fix number
            max_sampled = self.opt.max_depth # * 1.40
            depth_stride = stride / 2 # 采样翻倍

            N_step = int((max_sampled // depth_stride))

            pts_xyz = torch.from_numpy(pcd_down_sample_point).float()
            vector = pts_xyz - all_cam_center
            length = vector.norm(dim=1)
            norm_vector = vector / length[:, None]
            mean_step = torch.tensor([depth_stride]).repeat(pts_xyz.shape[0])

            train_rng = torch.arange(N_step)[None].float()

            # pdb.set_trace()
            origin = torch.from_numpy(all_cam_center).repeat(pts_xyz.shape[0], 1)
            instance_step = mean_step[:, None] * train_rng.repeat(pts_xyz.shape[0], 1)
            pts_xyz_all = origin[..., None, :] + norm_vector[..., None, :] * instance_step[..., None]


            # get mask
            depth_mask = self.get_mask(pts_xyz_all)
            valid_pt = pts_xyz_all[~depth_mask]

            inputs['val_depth_empty'] = valid_pt
            inputs['mask'] = depth_mask
            inputs['total_depth_empty'] = pts_xyz_all
            inputs['surface_point'] = pts_xyz[..., :]
            inputs['origin'] = origin



        # return

        # # for val label
        val_label = {}

        # for classification metric
        val_label['all_empty'] = inputs['all_empty']
        val_label['True_center'] = inputs['True_center']
        # val_label['True_voxel_group'] = inputs['True_voxel_group']

        # for depth metric
        val_label['val_depth_empty'] = inputs['val_depth_empty']
        val_label['mask'] = inputs['mask']
        val_label['total_depth_empty'] = inputs['total_depth_empty']
        val_label['surface_point'] = inputs['surface_point']
        val_label['origin'] = inputs['origin']

        # # save val label
        # DDAD
        # save_label_path = inputs['point_cloud_path'][0].replace('point_cloud_val', 'label/point_cloud_val_label_52_0.0_center_g_fix_num30_new')

        # nuscene
        # /data/ggeoinfo/Wanshui_BEV/data/nuscenes/point_cloud_full/samples/CAM_FRONT
        save_label_path = inputs['point_cloud_path'][0].replace('point_cloud_full', 'point_cloud_val_label/label_52_0.4_surface_fix_num30_depth_52')
        print(save_label_path)
        # exit()

        # (filepath, tempfilename) = os.path.split(save_label_path)
        # os.makedirs(filepath, exist_ok=True)
        # np.save(save_label_path, val_label)


    def get_occupancy_train_label(self, inputs):

        all_pose, pts_xyz = inputs['pose_spatial'], inputs['point_cloud'][0]

        # voxel_size = self.opt.max_depth/self.opt.voxels_size[0]  # 乘于2，再除于2，相消
        N_step = self.opt.N_trian

        mask1 = pts_xyz[:, 2] <= 0
        mask2 = pts_xyz[:, 2] >= self.opt.real_size[5]

        # print('self.opt.real_size[5]:', self.opt.real_size[5])

        xy_range = self.opt.max_depth

        # x
        mask3 = pts_xyz[:, 0] > xy_range
        mask4 = pts_xyz[:, 0] < -xy_range

        # y
        mask5 = pts_xyz[:, 1] > xy_range
        mask6 = pts_xyz[:, 1] < -xy_range


        # mask out the point cloud close to car, especially for nuscenes
        mask7 = (pts_xyz[:, 0] < 3.5) & (pts_xyz[:, 0] > -0.5)
        # y
        mask8 = (pts_xyz[:, 1] < 1.0) & (pts_xyz[:, 1] > -1.0)

        mask9 = mask7 & mask8

        mask = mask1 + mask2 + mask3 + mask4 + mask5 + mask6 + mask9

        GT_point = pts_xyz[~mask]
        pts_xyz = GT_point


        all_cam_center = torch.tensor([all_pose[:, 0, 3].mean(), all_pose[:, 1, 3].mean(), all_pose[:, 2, 3].mean()])
        pts_xyz = torch.from_numpy(pts_xyz).float()
        vector = pts_xyz - all_cam_center
        length = vector.norm(dim=1)
        norm_vector = vector / length[:, None]


        # mean_step = length/N_step
        stride = self.opt.val_reso
        last_empty_point = pts_xyz - stride * norm_vector
        empty_vector = last_empty_point - all_cam_center
        empty_length = empty_vector.norm(dim=1)
        mean_step = empty_length / N_step


        train_rng = torch.arange(N_step)[None].float()
        instance_step = mean_step[:, None] * train_rng.repeat(pts_xyz.shape[0], 1)
        pts_xyz_all = last_empty_point[..., None, :] - norm_vector[..., None, :]*instance_step[..., None]
        # 将pts_xyz 让有一个分辨率的位置，然后再取empty的点，包括测试也是


        all_empty = pts_xyz_all[:, 1:-3, :].flatten(0, 1)
        # car_box = all_cam_center[np.newaxis, :]
        # pcd_car = o3d.geometry.PointCloud()
        # pcd_car.points = o3d.utility.Vector3dVector(car_box)
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_car, voxel_size=1.5)
        # in_mask = voxel_grid.check_if_included(o3d.utility.Vector3dVector(all_empty))  # 可以将车放置一些点作为voxel
        # out_mask = ~np.array(in_mask)
        # all_empty = all_empty[out_mask]

        # label
        inputs['True_center'] = pts_xyz#.unsqueeze(0)
        inputs['all_empty'] = all_empty#.unsqueeze(0)

        return inputs



def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):

    with torch.no_grad():
        rays_o_all = torch.zeros(6, H, W, 3)
        rays_d_all = torch.zeros(6, H, W, 3)

        for i in range (6):
            # pdb.set_trace()
            # if K[i, ...] != K[i+6, ...] and c2w[i, ...] != c2w[i+6, ...]:
            #     print('false')

            rays_o, rays_d = get_rays(H, W, K[i, ...], c2w[i, ...], inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
            # 以上rays_o, rays_d 在车辆坐标系上，需要转到参考相机前视相机下

            rays_o_all[i,...] = rays_o
            rays_d_all[i,...] = rays_d

    return rays_o_all, rays_d_all


def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass


    elif mode == 'center':
        i, j = i + 0.5, j + 0.5


    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
    else:


        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1) #

    # Rotate ray directions from camera frame to the world frame

    # pdb.set_trace()
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d
