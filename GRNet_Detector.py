#! /usr/bin/python3
# -*- coding: utf-8 -*-

import os
from pprint import pprint
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import torch

from config import cfg
import utils.helpers
import utils.io
from models.grnet import GRNet


class GRNet_Detector:
    def __init__(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True

        self.n_points = 2048

        print('Use config:')
        pprint(cfg)

        self.grnet = GRNet(cfg)

        if torch.cuda.is_available():
            self.grnet = torch.nn.DataParallel(self.grnet).cuda()

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            print("GRNetDetector::load_model : model doesn't exists!")
            return False

        print('Recovering from %s ...' % (model_path))
        checkpoint = torch.load(model_path)
        self.grnet.load_state_dict(checkpoint['grnet'])

        self.grnet.eval()
        return True

    # For KITTI Dataset -> "/home/chli/github/GRNet/frame_0_car_0.txt"
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


    def detect_pcd(self, pcd_file_path):
        k = ['partial_cloud', 'bounding_box']
        v = [[], []]

        pcl = o3d.io.read_point_cloud(pcd_file_path)
        [x_min, y_min, z_min] = pcl.points[0]
        [x_max, y_max, z_max] = pcl.points[0]
        for point in pcl.points:
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

        pcl_np = np.asarray(pcl.points) - np.asarray([x_middle, y_middle, z_middle])

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
            return pointcloud_result


if __name__ == '__main__':
    model_path = "/home/chli/github/GRNet/GRNet-ShapeNet.pth"
    pcd_file_path = "/home/chli/github/GRNet/e431f79ac9f0266bca677733d59db4df.pcd"

    grnet_detector = GRNet_Detector()

    grnet_detector.load_model(model_path)

    pointcloud_result = grnet_detector.detect_pcd(pcd_file_path)

    output_folder = "/home/chli/github/gr-net/output/benchmark/02691156"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file_path = output_folder + '/test_out.h5'
    utils.io.IO.put(output_file_path, pointcloud_result)

    print('Test Output File = %s' % (output_file_path))

