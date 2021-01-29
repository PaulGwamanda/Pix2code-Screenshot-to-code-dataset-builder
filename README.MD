* `/dsl-library`
    * This is the location of the DSL library vocabulary and dataset builder. In this folder, you will be able to create your custom DSL library that will output two files: web-dsl-mapping.json and bootstrap.vocab which is required by the model to train on the dataset. More information on how to implement is  the README in that folder.
* `/datasets` 
    * This is the location of the datasets, the instructions on how to create a .gui file as well as examples are in this folder
* `/screenshot-to-code` 
    * This is the location of the training model (Keras version by Emil Walner), once the dataset and web-dsl-mapping files have been created, you can run this script to train the model. README file in folder
* `/API` 
    * This is the FLASK API to test your endpoints. Once your model is trained, you can use it to upload and image and retrieve the output in localhost/5000. Requires tensorflow to be locally installed. README in folder.
