from shutil import copyfile
from shutil import copytree, ignore_patterns
from distutils.dir_util import copy_tree
import os

png_pairs = "../png-pairs"
npz_pairs = "../npz-pairs"
npz_folder = os.listdir(npz_pairs)

# Create Gui files from create-gui-file.txt
print("Creating Gui files...")
List = open("create-gui-file.txt")
List2 = (s.strip() for s in List)
folder = "../png-pairs/"
# Loop through the list and create a file
for item in List2:
    open(folder + '/%s'%(item,), 'w')

copy_tree(png_pairs, npz_pairs)

for item in npz_folder:
    if item.endswith(".png"):
        os.remove(os.path.join(npz_pairs, item))

print("Complete!")