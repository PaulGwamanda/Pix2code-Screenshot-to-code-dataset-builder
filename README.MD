@todo: update Readme

Requires tensorflow on local machine, if have anaconda, run following command:

1. Activate tensorflow
- conda activate tensorflow

2. Follow the following steps: 
- Step 1: Run gulp in dsl_library
- Step 2: In /datasets, add custom images with naming convention (ie image-1, image-2)
- Step 3: Update txt file in /datasets/script/create-gui-file.txt following image naming convention (ie image-1.gui, image-3.gui)
- Step 4: Run create-gui-file.py once in /datasets/script/
- Step 4: Run compiler.py
- Step 5: Add gui descriptions to .gui files
- Step 6: Run bootstrap.py / generator.py 

Once sript has been run, all assets should be in training-assets folder
