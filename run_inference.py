#! /usr/bin/python3
# -*- coding: utf-8 -*-

from GRNet_Detector import GRNet_Detector

import os
import utils.io

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

