## Overview

This is the <b>custom dataset</b> builder, <b>web-dsl-mapping</b> and <b>bootstrap.vocab</b> generator for [Screenshot-2-Code](https://github.com/emilwallner/Screenshot-to-code) and [Pix2code](https://github.com/tonybeltramelli/pix2code).<br/> 

It's based off of the Pix2code library by Tony Tony Beltramelli, but in particular the screenshot-2-code version by Emil Walner which uses Keras + Tensorflow. 
 
It allows you to create your own custom datasets as well as vocab. 
It also includes training scripts for AWS, Colab, and local cpu. Folders: 




* `/dataset-builder` 
    * Build your dataset here, contains your .npz and .gui pairs for model training.
* `/dsl-builder`
    * This is where you build your dsl library (web-dsl-mapping.json + bootstrap.vocab). 
* `/training` 
    * Contains training scripts for Colab, AWS and local for model training   
    
My build: https://www.youtube.com/embed/Jr6DyjczXUA 
(Note the weights file is not included in this project).
