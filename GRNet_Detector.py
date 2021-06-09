#! /usr/bin/python3
# -*- coding: utf-8 -*-

import os
from pprint import pprint
import numpy as np
import open3d as o3d
import torch

from config import cfg
import utils.helpers
import utils.io
from models.grnet import GRNet


class GRNet_Detector:
    '''
    Init GRNet config and set torch env param
    Need to load model after create this class
    '''
    def __init__(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True

        self.n_points = 2048

        print('Use config:')
        pprint(cfg)

        self.grnet = GRNet(cfg)

        if torch.cuda.is_available():
            self.grnet = torch.nn.DataParallel(self.grnet).cuda()

    '''
    Input:
        model_path : .pth model path
    Return:
        load_success : whether load model success
    '''
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            print("GRNetDetector::load_model : model doesn't exists!")
            return False

        print('Recovering from %s ...' % (model_path))
        checkpoint = torch.load(model_path)
        self.grnet.load_state_dict(checkpoint['grnet'])

        self.grnet.eval()
        return True

    '''
    For KITTI Dataset -> "/home/chli/github/GRNet/frame_0_car_0.txt"

    Input:
        bbox_file : .txt bbox file path
    Return:
        bbox : np.array((8, 3), dtype=np.float32)
    '''
    def get_bbox(self, bbox_file):
        bbox_file_info = []
        with open(bbox_file, "r") as f:
            bbox_file_info = f.readlines()

        bbox = []

        for info in bbox_file_info:
            info_split = info.split("\n")[0].split(" ")
            temp_index = []
            for temp_info in info_split:
                temp_index.append(float(temp_info))
            bbox.append(temp_index)

        [x_min, y_min, z_min] = bbox[0]
        [x_max, y_max, z_max] = bbox[0]

        for i in range(1, len(bbox)):
            if bbox[i][0] < x_min:
                x_min = bbox[i][0]
            if bbox[i][1] < y_min:
                y_min = bbox[i][1]
            if bbox[i][2] < z_min:
                z_min = bbox[i][2]

            if bbox[i][0] > x_max:
                x_max = bbox[i][0]
            if bbox[i][1] > y_max:
                y_max = bbox[i][1]
            if bbox[i][2] > z_max:
                z_max = bbox[i][2]

        x_middle = (x_min + x_max) / 2.0
        y_middle = (y_min + y_max) / 2.0
        z_middle = (z_min + z_max) / 2.0

        for i in range(len(bbox)):
            bbox[i][0] -= x_middle
            bbox[i][1] -= y_middle
            bbox[i][2] -= z_middle

        bbox = np.asarray([bbox]).astype(np.float32)

        return bbox

    '''
    Input:
        pointcloud : [<=2048, 3]
    Return:
        pointcloud_result : np.array((16384, 3), dtype=np.float32)
    '''
    def detect(self, pointcloud):
        k = ['partial_cloud', 'bounding_box']
        v = [[], []]

        [x_min, y_min, z_min] = pointcloud[0]
        [x_max, y_max, z_max] = pointcloud[0]
        for point in pointcloud:
            if point[0] < x_min:
                x_min = point[0]
            if point[1] < y_min:
                y_min = point[1]
            if point[2] < z_min:
                z_min = point[2]

            if point[0] > x_max:
                x_max = point[0]
            if point[1] > y_max:
                y_max = point[1]
            if point[2] > z_max:
                z_max = point[2]

        x_middle = (x_min + x_max) / 2.0
        y_middle = (y_min + y_max) / 2.0
        z_middle = (z_min + z_max) / 2.0

        pcl_np = np.asarray(pointcloud) - np.asarray([x_middle, y_middle, z_middle])

        if pcl_np.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - pcl_np.shape[0], 3))
            pcl_np = np.concatenate([pcl_np, zeros])

        pcl_np = np.asarray([pcl_np]).astype(np.float32)
        v[0] = torch.from_numpy(pcl_np)

        #bbox = self.get_bbox("/home/chli/github/GRNet/frame_0_car_0.txt")
        #v[1] = torch.from_numpy(bbox)

        data = {}

        data[k[0]] = utils.helpers.var_or_cuda(v[0])
        #data[k[1]] = utils.helpers.var_or_cuda(v[1])

        with torch.no_grad():
            sparse_ptcloud, dense_ptcloud = self.grnet(data)

            pointcloud_result = dense_ptcloud.squeeze().cpu().numpy()
            print("return size : ", pointcloud_result.shape)
            return pointcloud_result

    '''
    Input:
        pcd_file_path : .pcd file path
    Return:
        pointcloud_result : np.array((16384, 3), dtype=np.float32)
    '''
    def detect_pcd_file(self, pcd_file_path):
        pcl = o3d.io.read_point_cloud(pcd_file_path)

        return self.detect(pcl.points)
    
if __name__ == '__main__':
    model_path = "/home/chli/github/GRNet/GRNet-ShapeNet.pth"
    pcd_file_path = "/home/chli/github/GRNet/e431f79ac9f0266bca677733d59db4df.pcd"

    grnet_detector = GRNet_Detector()

    grnet_detector.load_model(model_path)

    pointcloud_result = grnet_detector.detect_pcd_file(pcd_file_path)

    output_folder = "/home/chli/github/gr-net/output/benchmark/02691156"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file_path = output_folder + '/test_out.h5'
    utils.io.IO.put(output_file_path, pointcloud_result)

    print('Test Output File = %s' % (output_file_path))

