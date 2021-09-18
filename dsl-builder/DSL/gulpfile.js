'use strict';

var gulp = require('gulp');
var htmlToJson = require('../index.js');
var log = require('fancy-log');


/**
 * COMPILE AS JSON but you want to use your preferred filename
 * @return Json file
 */
gulp.task('default', function () {

    return gulp.src('./token-builder/token-builder.tpl')
        .pipe(htmlToJson({
            filename : 'web-dsl-mapping'
            , useAsVariable: false
            , isAngularTemplate : false
            , prefix : ""
        }))
        .pipe(gulp.dest('./web-dsl-mapping'))
        .on('error', log)
        ;
});
