#! /usr/bin/python3
# -*- coding: utf-8 -*-

from GRNetDetector.GRNet_Detector import GRNet_Detector

import os
from GRNetDetector.utils.io import IO

if __name__ == '__main__':
    model_path = "/home/chli/GRNet/GRNet-ShapeNet.pth"
    pcd_file_path = "/home/chli/GRNet/shapenet-20210607T072352Z-001/shapenet/test/partial/02691156/e431f79ac9f0266bca677733d59db4df.pcd"

    grnet_detector = GRNet_Detector()

    grnet_detector.load_model(model_path)

    pointcloud_result = grnet_detector.detect_pcd_file(pcd_file_path)

    output_folder = "/home/chli/github/gr-net/output/benchmark/02691156"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file_path = output_folder + '/test_out.h5'
    IO.put(output_file_path, pointcloud_result)

    print('Test Output File = %s' % (output_file_path))

