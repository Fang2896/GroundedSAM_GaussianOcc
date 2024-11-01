# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import configargparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
         
          self.parser = configargparse.ArgumentParser()

          self.parser.add_argument('--config', is_config_file=True,
                                   help='config file path',
                                   default = './configs/ddad_volume.txt')

          # PATHS
          self.parser.add_argument("--data_path",
                                   type=str,
                                   help="path to the training data",
                                   default=os.path.join(file_dir, "kitti_data"))

          self.parser.add_argument("--log_dir",
                                   type=str,
                                   help="log directory",
                                   default='./logs')


          # TRAINING options
          self.parser.add_argument("--model_name",
                                   type=str,
                                   help="the name of the folder to save the model in",
                                   default="ddad2023_rerun/1010_sdf")
          self.parser.add_argument("--split",
                                   type=str,
                                   help="which training split to use",
                                   choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                   default="eigen_zhou")
          self.parser.add_argument("--num_layers",
                                   type=int,
                                   help="number of resnet layers",
                                   default=34,
                                   choices=[18, 34, 50, 101, 152])
          self.parser.add_argument("--dataset",
                                   type=str,
                                   help="dataset to train on",
                                   default="kitti"
                                   )
          self.parser.add_argument("--png",
                                   help="if set, trains from raw KITTI png files (instead of jpgs)",
                                   action="store_true")
          self.parser.add_argument("--height",
                                   type=int,
                                   help="input image height",
                                   default=384)
          self.parser.add_argument("--width",
                                   type=int,
                                   help="input image width",
                                   default=640)


          self.parser.add_argument("--height_ori",
                                   type=int,
                                   help="original input image height",
                                   default=1216)
          self.parser.add_argument("--width_ori",
                                   type=int,
                                   help="original input image width",
                                   default=1936)
          self.parser.add_argument("--disparity_smoothness",
                                   type=float,
                                   help="disparity smoothness weight",
                                   default=0.001)

          self.parser.add_argument("--min_depth",
                                   type=float,
                                   help="minimum depth",
                                   default=1.0)
          self.parser.add_argument("--max_depth",
                                   type=float,
                                   help="maximum depth",
                                   default=52)
          self.parser.add_argument("--use_stereo",
                                   help="if set, uses stereo pair for training",
                                   action="store_true")
          self.parser.add_argument("--frame_ids",
                                   nargs="+",
                                   type=int,
                                   help="frames to load, currently only support for 3 frames",
                                   default=[0])


          self.parser.add_argument("--eval_only",
                                   help="if set, only evaluation",
                                   action="store_true")
          self.parser.add_argument("--use_fix_mask",
                                   help="if set, use self-occlusion mask (only for DDAD)",
                                   action="store_true")


          self.parser.add_argument("--spatial", type=lambda x: x.lower() == 'true', default=False,
                                   help="if set, use spatial photometric loss")



          self.parser.add_argument("--joint_pose",
                                   help="if set, use joint pose estimation",
                                   action="store_true")
          self.parser.add_argument("--model_type",
                                   type=str,
                                   default="unet")


          # self.parser.add_argument("--use_sfm_spatial",
          #                          help="if set, use sfm pseudo label",
          #                          action="store_true")

          self.parser.add_argument("--use_sfm_spatial", type=lambda x: x.lower() == 'true', default=False,
                                   help="if set, use sfm pseudo label")


          self.parser.add_argument("--thr_dis",
                                   type=float,
                                   help="epipolar geometry threshold",
                                   default=1.0)
          self.parser.add_argument("--match_spatial_weight",
                                   type=float,
                                   help="sfm pretraining loss weight",
                                   default=0.1)
          self.parser.add_argument("--spatial_weight",
                                   type=float,
                                   help="spatial photometric loss weight",
                                   default=0.06)
          self.parser.add_argument("--skip",
                                   help="if set, use skip connection in CVT",
                                   action="store_true")

          self.parser.add_argument("--focal", type=lambda x: x.lower() == 'true', default=True,
                                   help="if set, use sfm pseudo label")



          self.parser.add_argument("--focal_scale",
                                   type=float,
                                   help="the global focal length to normalize depth",
                                   default=500)

          # OPTIMIZATION options
          self.parser.add_argument("--batch_size",
                                   type=int,
                                   help="batch size",
                                   default=1)
          self.parser.add_argument("--B",
                                   type=int,
                                   help="real batch size",
                                   default=1)

          self.parser.add_argument("--learning_rate",
                                   type=float,
                                   help="learning rate",
                                   default=1e-4)




          self.parser.add_argument("--num_epochs",
                                   type=int,
                                   help="number of epochs",
                                   default=8)
          self.parser.add_argument("--scheduler_step_size",
                                   type=int,
                                   help="step size of the scheduler",
                                   default=6)

          # ABLATION options
          self.parser.add_argument("--v1_multiscale",
                                   help="if set, uses monodepth v1 multiscale",
                                   action="store_true")
          self.parser.add_argument("--avg_reprojection",
                                   help="if set, uses average reprojection loss",
                                   action="store_true")
          self.parser.add_argument("--disable_automasking",
                                   help="if set, doesn't do auto-masking",
                                   action="store_true")
          self.parser.add_argument("--predictive_mask",
                                   help="if set, uses a predictive masking scheme as in Zhou et al",
                                   action="store_true")
          self.parser.add_argument("--no_ssim",
                                   help="if set, disables ssim in the loss",
                                   action="store_true")
          self.parser.add_argument("--weights_init",
                                   type=str,
                                   help="pretrained or scratch",
                                   default="pretrained",
                                   choices=["pretrained", "scratch"])
          self.parser.add_argument("--pose_model_input",
                                   type=str,
                                   help="how many images the pose network gets",
                                   default="pairs",
                                   choices=["pairs", "all"])
          self.parser.add_argument("--pose_model_type",
                                   type=str,
                                   help="normal or shared",
                                   default="separate_resnet")


          # SYSTEM options
          self.parser.add_argument("--no_cuda",
                                   help="if set disables CUDA",
                                   action="store_true")
          self.parser.add_argument("--num_workers",
                                   type=int,
                                   help="number of dataloader workers",
                                   default=20)

          # LOADING options
          self.parser.add_argument("--load_weights_folder",
                                   type=str,
                                   help="name of model to load",
                                   default="ddad2023_rerun/903_sdf_all_volume_True_loss_gt_epoch_12_r_No_ms_False_grid_bilinear/method_rendering_sky_No_sdf_1_gtpos_False_uset_No_meas_abs_abs_False_focal_False_semiw_2.0_sm_2.0_sp_False_val_0.4_voxel_No_sur_0.5_empty_w_10.0_depth_52.0_out_1_en_50_input_64_vtrans_simple/step_0.5_size_256_rlde_0.001_aggregation_3dcnn_type_density_pe_No/models/weights_18481")
          self.parser.add_argument("--models_to_load",
                                   nargs="+",
                                   type=str,
                                   help="models to load",
                                   default=["encoder", "depth"])

          # LOGGING options
          self.parser.add_argument("--log_frequency",
                                   type=int,
                                   help="number of batches between each tensorboard log",
                                   default=25)
          self.parser.add_argument("--save_frequency",
                                   type=int,
                                   help="number of epochs between each save",
                                   default=1)
          self.parser.add_argument("--eval_frequency",
                                   type=int,
                                   help="number of epochs between each save",
                                   default=1000)


          # EVALUATION options
          self.parser.add_argument("--eval_stereo",
                                   help="if set evaluates in stereo mode",
                                   action="store_true")
          self.parser.add_argument("--eval_mono",
                                   help="if set evaluates in mono mode",
                                   action="store_true")
          self.parser.add_argument("--disable_median_scaling",
                                   help="if set disables median scaling in evaluation",
                                   action="store_true")
          self.parser.add_argument("--pred_depth_scale_factor",
                                   help="if set multiplies predictions by this number",
                                   type=float,
                                   default=1)
          self.parser.add_argument("--ext_disp_to_eval",
                                   type=str,
                                   help="optional path to a .npy disparities file to evaluate")
          self.parser.add_argument("--eval_split",
                                   type=str,
                                   default="eigen",
                                   choices=[
                                        "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                   help="which split to run eval on")
          self.parser.add_argument("--save_pred_disps",
                                   help="if set saves predicted disparities",
                                   action="store_true")
          self.parser.add_argument("--no_eval",
                                   help="if set disables evaluation",
                                   action="store_true")
          self.parser.add_argument("--eval_eigen_to_benchmark",
                                   help="if set assume we are loading eigen results from npy but "
                                        "we want to evaluate using the new benchmark.",
                                   action="store_true")
          self.parser.add_argument("--eval_out_dir",
                                   help="if set will output the disparities to this folder",
                                   type=str)
          self.parser.add_argument("--post_process",
                                   help="if set will perform the flipping post processing "
                                        "from the original monodepth paper",
                                   action="store_true")

          self.parser.add_argument("--local_rank", default=0, type=int)


          # customized
          self.parser.add_argument("--volume_depth",
                                   type=lambda x: x.lower() == 'true', default=False,
                                   help="if set, using the depth from volume rendering, rather than the depthdecoder",
                                   )

          self.parser.add_argument("--loss_type",  type=str,
                                   help="the loss for training [self, semi, gt]",
                                   default='self')


          self.parser.add_argument("--trans2voxel", type=str,
                    help="the manner from 2d feature to 3D voxel:[interpolation, transformer]",
                    default = "interpolation")


          self.parser.add_argument("--voxels_size", type=int, action='append', default=[16, 256, 256],
                              help='the resolution of the voxel for rendering: Z, Y, X = 200, 8, 200')

          self.parser.add_argument("--real_size", type=float, action='append', default=[0, 52, 0, 52, -1, 6.0],
                                   help='the real scale of the voxel: XMIN, XMAX, ZMIN, ZMAX, YMIN, YMAX')

          self.parser.add_argument("--scales", action='append', type=int,
                                   help="scales used in the loss",
                                   # default: [0, 1, 2, 3]
                                   default=[])

          self.parser.add_argument("--stepsize",
                                   help="stepsize for rendering",
                                   type=float,
                                   default=0.5)


          self.parser.add_argument("--en_lr",
                                   type=float,
                                   help="learning rate for encoder in volume rendering",

                                   default=0.0001)

          self.parser.add_argument("--de_lr",
                                   type=float,
                                   help="learning rate for decoder (3D CNN) in volume rendering",

                                   default=0.001)


          self.parser.add_argument("--aggregation", type=str,
                                   help="the type of the feature aggregation [mlp 3dcnn 2dcnn]",
                                   default="3dcnn"
                                   )

          self.parser.add_argument("--tvloss", type=float, default=0.0,
                                   help="if tvloss > 0, using the  tv loss on the final voxel grid",)

          self.parser.add_argument("--pose_aug",
                                   type=str,
                                   help="do the augmentation on the camera pose for image level or car level [image, car, No]",
                                   default='No')

          self.parser.add_argument("--render_type",
                                   type=str,
                                   help="rednering by the density or probability [density, prob]",
                                   default='density')

          self.parser.add_argument("--position",
                                   type=str,
                                   help="rednering by the density or probability [embedding, encoding]",
                                   default='No')

          self.parser.add_argument("--data_type",
                                   type=str,
                                   help=" data size for traing and testing - > [train_all, all, mini, tiny]",
                                   default='all')

          self.parser.add_argument("--grad_acc", type=int, default=0,
                                   help="if grad_acc > 0, using the gradient accumulation", )

          self.parser.add_argument("--semi_weight", type=float, default=20.0,
                                   help="weight to balance the gtloss and self supervision loss during the semi supervision", )

          self.parser.add_argument("--alpha_init", type=float, default=1,
                                   help="weight to balance the gtloss and self supervision loss during the semi supervision", )

          self.parser.add_argument("--aug_size", type=float, default=0.3, help="the size of the pose augmentation", )

          self.parser.add_argument("--log", type=lambda x: x.lower() == 'true', default = False,
                                   help="if set, using line space sample")

          self.parser.add_argument("--render_h",
                                   type=int,
                                   help="input image height",
                                   default=224)
          self.parser.add_argument("--render_w",
                                   type=int,
                                   help="input image width",
                                   default=352)

          self.parser.add_argument("--cam_aug", type=lambda x: x.lower() == 'true', default=False, help="if set, using camera augmentation")

          self.parser.add_argument("--view_trans",
                                   type=str,
                                   help="the manner for image space to 3D volume space [simple, lift, bevformer]",
                                   default='simple')

                                   
          self.parser.add_argument("--input_channel",
                                   type=int,
                                   help="the final feature channel in the encoder",
                                   default=64)

          self.parser.add_argument("--con_channel",
                                   type=int,
                                   help="the final feature channel in the encoder",
                                   default=16)

          self.parser.add_argument("--out_channel",
                                   type=int,
                                   help="the output channel of the voxel",
                                   default=1)

          self.parser.add_argument("--em_tv", type=lambda x: x.lower() == 'true', default=False,
                                   help="if set, using line space sample")

          self.parser.add_argument("--cam_N", type=int, help="THE NUM OF CAM", default=6)

          self.parser.add_argument("--method",
                                   type=str,
                                   help="the method for the comparison [surrounddepth, CRF, monodepth2]",
                                   default='large07')

          self.parser.add_argument("--upsample", type=lambda x: x.lower() == 'true', default=False,
                                   help="if set, using line space sample")

          self.parser.add_argument("--encoder",
                                   type=str,
                                   help="the method for the comparison [101, 50, sw]",
                                   default='large07')

          self.parser.add_argument("--actfun",
                                   type=str,
                                   help="activation in the decoder [Softplus, ReLU, SiLU]",
                                   default='ReLU')

          self.parser.add_argument("--loss",
                                   type=str,
                                   help="activation in the decoder [l1, sml1, silog, rl1]",
                                   default='silog')

          self.parser.add_argument("--evl_score", type=lambda x: x.lower() == 'true', default=False,
                                   help="if set, eval the occupancy score!")

          self.parser.add_argument("--surfaceloss", type=float, default=0.01,
                                   help="if tvloss > 0, using the  surface loss", )

          self.parser.add_argument("--empty_w", type=float, default=0.01,
                                   help="the weight of the empty point loss for the l1 grid loss", )

          self.parser.add_argument("--l1_voxel", type=str,
                                   help="activation in the decoder [No, ce, l1, ce_only, l1_only]", default='No')

          self.parser.add_argument("--val_reso", type=float, default=0.4, help="the resolution of the voxel in the evaluation [0.2, 0.4]")

          self.parser.add_argument("--N_trian", type=int, help="THE NUM OF sample point in the voxel training", default=30)
          self.parser.add_argument("--center_label", type=lambda x: x.lower() == 'true', default=False, help="if set, directly use the center of the voxel!")
          self.parser.add_argument("--val_depth", type=lambda x: x.lower() == 'true', default=True, help="if set, do the depth voxel evaluation!")
          self.parser.add_argument("--use_t",  type=str, default='No', help="if [No, 2d, 3d], do the temporal information fusion!")

          self.parser.add_argument("--detach", type=lambda x: x.lower() == 'true', default=False, help="if set, do the depth voxel evaluation!")
          self.parser.add_argument("--surround_view", type=lambda x: x.lower() == 'true', default=False, help="if set, eval the surrounding view depth!")

          # self.parser.add_argument("--pretrain_path", type=str, help="pretrain path for the encoder", default='/home/wsgan/project/bev/SurroundDepth/networks/sceneflow.ckpt')
          self.parser.add_argument("--pretrain_path", type=str, help="pretrain path for the encoder",
                                   default='No')

          self.parser.add_argument("--val_g", type=lambda x: x.lower() == 'true', default=False, help="if set, eval the surrounding view depth!")

          self.parser.add_argument("--ground_prior", type=lambda x: x.lower() == 'true', default=False,  help="if set, set the ground as 1!")

          self.parser.add_argument("--downsample_val", type=lambda x: x.lower() == 'true', default=False, help="if set")

          self.parser.add_argument("--val_binary", type=lambda x: x.lower() == 'true', default=False, help="if set")

          self.parser.add_argument("--abs", type=lambda x: x.lower() == 'true', default=False, help="if set, use the abs scale")

          self.parser.add_argument("--gt_pose", type=str, default='No', help="if set, use the gt pose")

          self.parser.add_argument("--dataroot", type=str, help="the root for the ddad and nuscenes dataset", default='/data/ggeoinfo/Wanshui_BEV/data/ddad')

          self.parser.add_argument("--mask_img", type=lambda x: x.lower() == 'true', default=False, help="if set, use the gt pose")

          self.parser.add_argument("--sky_loss", type=lambda x: x.lower() == 'true', default=False, help="if set, use the sky mask prior")

          self.parser.add_argument("--similarity", type=str, default='cat1', help="if [No, 2d, 3d], do the temporal information fusion!")

          self.parser.add_argument("--refine", type=str, default='No', help="if set, use the 2D CNN to refine the predicted disp")

          self.parser.add_argument("--multi_scale", type=lambda x: x.lower() == 'true', default=False, help="if set, use the sky mask prior")

          self.parser.add_argument("--grid_sample", type=str, default='bilinear', help="if set, bilinear, nearest")

          self.parser.add_argument("--model_sky", type=str, default='No', help="if set, model the sky with mlp")

          self.parser.add_argument("--sdf", type=str, default='No', help="if set, model the sky with mlp")

          self.parser.add_argument("--vis_sdf", type=lambda x: x.lower() == 'true', default=False, help="if set, vis sdf")

          self.parser.add_argument("--density_0", type=lambda x: x.lower() == 'true', default=True, help="if set, use the old rendering code")

          self.parser.add_argument("--beta", type=float, default=1.0, help="the initial weight of beta in sdf")

          self.parser.add_argument("--rgb_loss", type=float, default=0.05, help="the loss weight for the rendering rgb")

          self.parser.add_argument("--novel_view_loss", type=lambda x: x.lower() == 'true', default=False, help="if set, use the novel view loss including depth, rgb")

          self.parser.add_argument("--val_novel_view", type=lambda x: x.lower() == 'true', default=False, help="if set, val novel view score")

          self.parser.add_argument("--sampled_rgb", type=lambda x: x.lower() == 'true', default=False, help="if set, sampled rgb from img")

          self.parser.add_argument("--flow", type=str, default='No', help="if set, model the motion with flow field")

          self.parser.add_argument("--optflow", type=str, default='No', help="if set, model the motion with flow field")

          self.parser.add_argument("--motion_scale", type=float, default=1.0, help="the initial weight of motion")

          self.parser.add_argument("--pseudo_depth", type=str, default='No', help="if set, using the scale ambiguity loss from the dt depth or pseudo depth")


          self.parser.add_argument("--gs", type=str, default='No', help="if set, using gs rendering!")

          self.parser.add_argument("--gs_scale", type=float, default=0.01, help="the scale of gs")

          self.parser.add_argument("--mask_val", type=float, default=0, help="the mask of  gs actiation")

          self.parser.add_argument("--spring", type=int, default=0, help="if set, using spring offset in gs!")

          self.parser.add_argument("--enl", type=float, default=0.0, help="the weight of the entropy loss for the guassian opacity!")



          # raft flow
          # /home/wsgan/project/bev/SurroundingNeRF/third_party/flow/RAFT/models/raft-kitti.pth
          # /home/wsgan/project/bev/SurroundingNeRF/third_party/flow/RAFT/models/raft-things.pth
          self.parser.add_argument('--model', type=str, default='/home/wsgan/project/bev/SurroundingNeRF/third_party/flow/RAFT/models/raft-things.pth', help="restore checkpoint")
          # self.parser.add_argument('--model', type=str, default='/home/wsgan/project/bev/SurroundingNeRF/third_party/flow/RAFT/models/raft-kitti.pth', help="restore checkpoint")

          self.parser.add_argument('--path', help="dataset for evaluation")
          self.parser.add_argument('--small', action='store_true', help='use small model')
          self.parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
          self.parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

          self.parser.add_argument('--eval', default = 'kitti_validation', help='eval benchmark')

          self.parser.add_argument("--pseudo_lidar", type=str, default='No', help="if set, using depth map to generate the pseudo lidar point!")

          self.parser.add_argument("--gs_depth", type=str, default='yes', help="if set, using depth map to generate the pseudo lidar point!")

          self.parser.add_argument("--alignment_loss", type=float, default=1.0, help="if set, using depth map to generate the pseudo lidar point!")

          self.parser.add_argument("--temporal_loss", type=float, default=0, help="if set, using depth map to generate the pseudo lidar point!")

          self.parser.add_argument("--o_mask", type=str, default='No', help="if set, using depth map to generate the pseudo lidar point!")

          self.parser.add_argument("--detach_pose", type=str, default='No', help="if set, using depth map to generate the pseudo lidar point!")

          self.parser.add_argument("--stage", type=int, default=1, help="if set, use the stage one or two of the depth!")

          # self.parser.add_argument("--stage", type=int, default=0, help="if set, use the stage one or two of the depth!")

          self.parser.add_argument("--min_depth_test", type=float, help="the min depth for the evaluation", default=0.1)

          self.parser.add_argument("--gs_prior", type=str, default='No', help="if set, using gs prior")

          self.parser.add_argument("--gs_proj", type=str, default='No', help="if set, using gs prior")

          self.parser.add_argument("--suspend_list", action='append', type=int,
                                   help="scales used in the loss", default=[15, 16])

          self.parser.add_argument("--erode_kernal", type=int, help="the erode_kernal size", default=0)
          
          self.parser.add_argument("--scale_2d", type=float, default=0.02, help="the scale of gs")


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
