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

## Training

There are two ways to train the model: 
> - <b>bootstrap.py</b>: loads the entire dataset as one batch (uses lots of memory)
> - <b>bootstrap_generator.py</b>: loads the dataset in batches (uses less memory)

To train the model, use either of these and run:

```
python train_local-bootstrap.py 
```
This will generate your <b>model.json</b> and save your weights to <b>weights.hdf5</b> format.  <br>
You will also see the model training in your terminal as well as epochs.

## Test model accuracy

To test your trained output, open the test_model_accuracy.py file and run:
>
```
python test_model_accuracy.py 
```
> *To improve accuracy of unseen examples, train the model on a larger dataset. <br>

## Compile to website template
> To compile the model output to a responsive bootstrap build that is deployable, head over to the weights/eval/api folder and follow the readme instructions there:


<br><b>
NOTE: 
The original model was trained on powerful GPUs, so you may run out of memory training on local CPU/GPU if your dataset is too large.   
</b>

