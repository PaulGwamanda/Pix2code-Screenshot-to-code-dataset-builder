## Overview

Before running the commands below, <b>make sure you have created the web-dsl-mapping.json first</b> file (navigate to /dsl-builder follow the readme instructions), once done:

Requires <b>Anaconda</b> with installed packages: 
- Tensorflow ^2.1
- Keras ^2.3.1
- py-opencv ^33.4 
- Numpy ^1.18)

## Installation

> Activate Anaconda tensorflow:
>
```
conda activate tensorflow
```
> Run preprocessor to create bootstrap.vocab file + npz pairs:
>
```
python preprocessor.py  
```
> You can now run the local training scripts to start train the model:
>
```
python train.py 
```
This will generate your <b>model.json</b> and save your weights to <b>weights.hdf5</b> format.  <br>

## Test model accuracy

> To test your trained output, open the test_model_accuracy.py file and run:
>
```
python test_model_accuracy.py 
```
> *To improve accuracy of unseen examples, train the model on a larger dataset. <br>

## Compile to website template
> To compile the model output to a responsive bootstrap build that is deployable, head over to the compiler/api folder and follow the readme instructions there:


<br><b>
NOTE: 
The original model was trained on powerful GPUs, so you may run out of memory training on local CPU/GPU if your dataset is too large.   
</b>

