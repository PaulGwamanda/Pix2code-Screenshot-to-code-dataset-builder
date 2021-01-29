## Overview

This project is based off of the Pix2code library, particulalry the [screenshot-2-code](https://github.com/emilwallner/Screenshot-to-code) version by Emil Walner which uses Keras. <br/><br/>
The project has been customised to include the creation of your own DSL vocab library (**web-dsl-mapping.json**), **bootstrap.vocab** file for the tokens, and **.GUI datasets** structure as well as examples. <br/><br/>
Below is the list of the folders in the repository as well as their use case. <br/>

* `/dsl-library`
    * This is the location of the DSL library vocabulary and dataset builder. In this folder, you will be able to create your custom DSL library that will output two files: web-dsl-mapping.json and bootstrap.vocab which is required by the model to train on the dataset. More information on how to implement is  the README in that folder.
* `/datasets` 
    * This is the location of the datasets, the instructions on how to create a .gui file as well as examples are in this folder
* `/screenshot-to-code` 
    * This is the location of the training model (Keras version by Emil Walner), once the dataset and web-dsl-mapping files have been created, you can run this script to train the model. README file in folder
* `/API` 
    * This is the FLASK API to test your endpoints. Once your model is trained, you can use it to upload and image and retrieve the output in localhost/5000. Requires tensorflow to be locally installed. README in folder.

* `/website` (optional)
    * This is the angular 8 version of the site which consumes the endpint created by FLASK. You can upload an image here and it will fetch the corresponding code from the API.
