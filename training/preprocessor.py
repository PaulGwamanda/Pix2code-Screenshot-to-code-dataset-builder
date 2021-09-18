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

import glob
import json
import os
from shutil import copyfile

IMAGE_SIZE = 256

dsl_mapping = '../dsl-builder/DSL/web-dsl-mapping/web-dsl-mapping.json'
bootstrap_vocab = './compiler/assets/bootstrap.vocab'
read_files = glob.glob('./dataset/png-pairs/*.gui')
special_tokens = './compiler/assets/bootstrap.vocab'
dsl_file_path = '../dsl-builder/DSL/web-dsl-mapping/web-dsl-mapping.json'
dsl_final_output = 'compiler/assets/web-dsl-mapping.json'
png_pairs = 'dataset/png-pairs'
npz_pairs = 'dataset/npz-pairs'
train_path = 'dataset/train/'


# If file exists, delete these files before pre-processing
if os.path.isfile(bootstrap_vocab):
    os.remove(bootstrap_vocab)

# Get all files from datasets/gui-files and create single file with all then tokens
def unique_file(input_filename, output_filename):
    print("Generating bootstrap.vocab file...")

    input_file = open(input_filename, 'r')

    file_contents = input_file.read()
    input_file.close()
    word_list = file_contents.split()

    file = open(output_filename, 'w')

    unique_words = set(word_list)
    for word in unique_words:
        file.write(str(word) + " \n")

    file.close()

# Strip and pull out only unique words from above list
tokens = './compiler/assets/tokens.txt'
tokenKeysFromDataset = './compiler/assets/token-keys-from-dataset.txt'

with open(tokens, "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())

 # Remove unnecessary characters from the words (ie: },{, and comma
    cleaned_file = "./compiler/assets/cleaned_file.txt"

    delete_list = ["}", "{", ","]
    fin = open(tokens)
    fout = open(cleaned_file, "w+")
    for line in fin:
        for word in delete_list:
            line = line.replace(word, "")
        fout.write(line)
    fin.close()
    fout.close()

unique_file('./compiler/assets/cleaned_file.txt', tokenKeysFromDataset)

# Get all keys from web-dsl-mapping.json
with open(dsl_mapping) as json_file:
    dsl_mapping = json.load(json_file)
    for i in dsl_mapping:
        f = open(bootstrap_vocab, "a")
        f.write(i + ', ')

os.remove(tokens)
os.remove(tokenKeysFromDataset)
os.remove(cleaned_file)


# convert PNG images to numpy arrays step 1
print("Converting dataset to numpy arrays...")
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

# Create an empty special_tokens file
f = open(special_tokens, 'a+')
f.write(' ')
f.close()

# Find all '.gui' file extensions in data folder and combine them
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
print("Generating web-dsl-mapping.json file...")
shutil.copyfile(dsl_file_path, dsl_final_output)

# Get all keys from web-dsl-mapping.json
print("Saving all assets...")

# Opening JSON file
with open(dsl_file_path ) as json_file:
    dsl_file_path = json.load(json_file)

    for i in dsl_file_path:
        f = open(bootstrap_vocab, "a")
        f.write(i + ' \n')

print('All assets saved in /compiler/assets folder!')
