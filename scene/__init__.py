#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from utils.system_utils import mkdir_p

class Scene:
    """b
     Scene 类用于管理场景的3D模型,包括相机参数、点云数据和高斯模型的初始化和加载
    """
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path # 模型文件保存路径
        self.loaded_iter = None # 已加载的迭代次数
        self.gaussians = gaussians # 高斯模型对象
        # 检查并加载已有的训练模型
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {} # 用于训练的相机参数
        self.test_cameras = {} # 用于测试的相机参数

        # 根据数据集类型（COLMAP或Blender）加载场景信息 读取相机内外参构建SceneInfo变量
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif 'zju_mocap_refine' in args.source_path: 
            if os.path.exists(os.path.join(args.source_path, "annots.npy")):
                print("Found annots.json file, assuming ZJU_MoCap_refine data set!,path: "+os.path.join(args.source_path, "annots.npy"))
                scene_info = sceneLoadTypeCallbacks["ZJU_MoCap_refine"](args.source_path, args.white_background, args.exp_name, args.eval)
        elif 'monocap' in args.source_path:
            print("assuming MonoCap data set!")
            scene_info = sceneLoadTypeCallbacks["MonoCap"](args.source_path, args.white_background, args.exp_name, args.eval)
        elif 'dna_rendering' in args.source_path:
            print("assuming dna_rendering data set!")
            scene_info = sceneLoadTypeCallbacks["dna_rendering"](args.source_path, args.white_background, args.exp_name, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 构建train_cameras和test_cameras
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 初始化高斯点云。如果是初次训练，初始化3D高斯模型；否则，加载已有模型
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            #从点云数据初始化模型参数 走这个
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent,args.atom_init_quantile)

        if self.gaussians.motion_offset_flag:
            model_path = os.path.join(self.model_path, "mlp_ckpt", "iteration_" + str(self.loaded_iter), "ckpt.pth")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location='cuda:0')
                self.gaussians.pose_decoder.load_state_dict(ckpt['pose_decoder'])
                self.gaussians.lweight_offset_decoder.load_state_dict(ckpt['lweight_offset_decoder'])

    # 模型的保存：将场景模型的高斯模型导出为ply文件
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        if self.gaussians.motion_offset_flag:
            model_path = os.path.join(self.model_path, "mlp_ckpt", "iteration_" + str(iteration), "ckpt.pth")
            mkdir_p(os.path.dirname(model_path))
            torch.save({
                'iter': iteration,
                'pose_decoder': self.gaussians.pose_decoder.state_dict(),
                'lweight_offset_decoder': self.gaussians.lweight_offset_decoder.state_dict(),
            }, model_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]