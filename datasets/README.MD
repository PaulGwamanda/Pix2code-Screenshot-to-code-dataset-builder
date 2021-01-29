## Overview

Lorem ipsum dolar sit amet:

* `/datasets`
    * This is the location of the DSL library vocabulary and dataset builder. In this folder, you will be able to create your custom DSL library that will output two files: web-dsl-mapping.json and bootstrap.vocab which is required by the model to train on the dataset. More information on how to implement is  the README in that folder.
    * should use only if you have 1 group template.
* `useAsVariable` (optional)
    * default false
    * If set to true, it will output your file as a javascript variable. Otherwise, json file
* `isAngularTemplate` (optional)
    * default false
    * If set to true, it will output your file as an angular template cache.
* `prefix` (optional)
    * set the prefix on your angular module name

* `includePath` (optional)
    * Takes a String or an Array of paths.
    * If set, gulp-html-to-json will use these folders as base path when searching for files.
