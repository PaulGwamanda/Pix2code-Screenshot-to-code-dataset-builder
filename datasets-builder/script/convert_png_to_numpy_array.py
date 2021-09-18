from __future__ import print_function
from __future__ import absolute_import
from distutils.dir_util import copy_tree

import os
import sys
import glob
import json
import re
import shutil
from shutil import copytree, ignore_patterns
import numpy as np

IMAGE_SIZE = 256
input_path = ''
output_path = ''

class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")

print("Converting images to numpy arrays...")

for f in os.listdir(input_path):
    if f.find(".png") != -1:
        img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)
        file_name = f[:f.find(".png")]

        np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
        retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

        assert np.array_equal(img, retrieve)

        shutil.copyfile("{}/{}.gui".format(input_path, file_name), "{}/{}.gui".format(output_path, file_name))

print("Numpy arrays saved in {}".format(output_path))