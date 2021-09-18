from __future__ import print_function
from __future__ import absolute_import
from distutils.dir_util import copy_tree

import glob
import shutil
from shutil import copytree, ignore_patterns
import numpy as np
import os, fnmatch
import os, sys
import json
import glob
import json
import os
from shutil import copyfile


IMAGE_SIZE = 256

special_tokens = "training-assets/special_tokens.txt"
dsl_file_path = "dsl-builder/DSL/web-dsl-mapping/web-dsl-mapping.json"
dsl_final_output = "compiler/assets/web-dsl-mapping.json"
png_pairs = "dataset/png-pairs"
npz_pairs = "dataset/npz-pairs"
# html_folder = "dsl-library/DSL/templates"
train_path = 'dataset/train/'
bootstrap_vocab = 'compiler/assets/bootstrap.vocab'

print("Converting images to numpy arrays...")

# convert PNG images to numpy arrays step 1
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

# Convert images to numpy arrays step 2
for f in os.listdir(png_pairs):
    if f.find(".png") != -1:
        img = Utils.get_preprocessed_img("{}/{}".format(png_pairs, f), IMAGE_SIZE)
        file_name = f[:f.find(".png")]
        np.savez_compressed("{}/{}".format(npz_pairs, file_name), features=img)
        retrieve = np.load("{}/{}.npz".format(npz_pairs, file_name))["features"]
        assert np.array_equal(img, retrieve)
        shutil.copyfile("{}/{}.gui".format(png_pairs, file_name), "{}/{}.gui".format(npz_pairs, file_name))


def findReplace(directory, find, replace, filePattern):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)


# Clean all files in png_pairs
findReplace(npz_pairs, "{", " { ", "*.gui")
findReplace(npz_pairs, "}", " } ", "*.gui")

print("Creating an empty special_tokens file...")

# Create an empty special_tokens file
f = open(special_tokens, 'a+')
f.write(' ')
f.close()

# Find all '.gui' file extensions in data folder and combine them
print('Cloning all relevant files...')
with open(special_tokens, 'wb') as outfile:
    for gui_tokens in glob.glob(npz_pairs + '/*/*.gui'):
        with open(gui_tokens, 'rb') as readfile:
            shutil.copyfileobj(readfile, outfile)

source = npz_pairs
if os.path.exists(train_path):
    shutil.rmtree(train_path)

# Copy files from png-pairs folder to npz-pairs folder
copytree(source, train_path, ignore=ignore_patterns('*.png'))

# Copy dsl mapping file to new file path
print("Creating web-dsl json file...")
shutil.copyfile(dsl_file_path, dsl_final_output)

# Get all keys from web-dsl-mapping.json
print("Creating bootstrap.vocab file...")

# Opening JSON file
with open(dsl_file_path ) as json_file:
    dsl_file_path = json.load(json_file)

    for i in dsl_file_path:
        f = open(bootstrap_vocab, "a")
        f.write(i + ' \n')

# Clean old files
print("Cleaning up...")
os.remove(special_tokens)

# Add div classes to empty files in DSL mapping
# for html_snippet in glob.glob(html_folder + '/*/*.html'):
#     if os.stat(html_snippet).st_size == 0:
#         with open(html_snippet, "w") as f:
#             f.write('<div>{}</div>')
print('Complete! Assets saved in /training-assets folder')
