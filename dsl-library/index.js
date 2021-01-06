"use strict";


var fs      = require("fs"),
    path    = require("path"),
    es      = require("event-stream"),
    plugError = require('plugin-error'),
    replaceExt = require('replace-ext'),
    glob    = require('glob'),
    frontMatter = require('front-matter'),

    _DIRECTIVE_REGEX = /^(.*=\s*([\w\.\*\/-]+)\s*:\s*([\w\.\*\/-]+\.html?\s*))$/gm
;

/**
 *
 * @param file
 * @returns {HTMLElement}
 * @private
 */
function _parse(file){

    if (fs.existsSync(file)) {
        var bufferContents = fs.readFileSync(file),
            parsed = frontMatter(bufferContents.toString().replace(/\s+/g, ' '));

        return parsed.body;

    } else {
        throw new plugError('gulp-html-to-json', 'File not found: ' + fullPath);
    }

}

var hasOwnProperty = Object.prototype.hasOwnProperty;

function isEmpty(obj) {

    // null and undefined are "empty"
    if (obj === null || typeof obj === 'undefined') return true;

    // Assume if it has a length property with a non-zero value
    // that that property is correct.
    if (obj.length > 0)    return false;
    if (obj.length === 0)  return true;

    // If it isn't an object at this point
    // it is empty, but it can't be anything *but* empty
    // Is it empty?  Depends on your application.
    if (typeof obj !== "object") return true;

    // Otherwise, does it have any properties of its own?
    // Note that this doesn't handle
    // toString and valueOf enumeration bugs in IE < 9
    for (var key in obj) {
        if (hasOwnProperty.call(obj, key)) return false;
    }

    return true;
}

/**
 *
 * @param dirname
 * @param p
 * @return {*}
 */
function indName (dirname, p) {

    if (!isEmpty(p) && !isEmpty(p.filename)) {

        return p.filename;
    }

    var n = dirname.split('.');

    n.pop();

    var nLast = n.pop(), nArr = [];

    if (nLast.indexOf('/') >= 0) {

        nArr = nLast.split('/')

    } else {

        nArr = nLast.split('\\')
    }

    return nArr.pop();
}

/**
 *
 * @param path
 * @param fname
 * @param useAsVariable
 * @returns {*}
 */
function replaceFilename (path, fname, useAsVariable) {

    var uvar = useAsVariable || false;

    var filename = path.replace(/^.*(\\|\/|\:)/, '');
    var nFname = path.replace(filename, fname);
    var ext = (uvar) ? ".js" : '.json';
    return replaceExt(nFname,  ext);
}

/**
 * Control HTML to JSON
 * @param fileContents
 * @param filePath
 * @param output
 * @param p
 */
function htmlToJsonController (fileContents, filePath, output, p) {

    var matches;
    var includePaths = false;

    while (matches = _DIRECTIVE_REGEX.exec(fileContents)) {

        var relPath     = path.dirname(filePath),
            fullPath    = path.join(relPath,  matches[3].replace(/['"]/g, '')).trim(),
            jsonVar     = matches[2],
            extension   = matches[3].split('.').pop();

        var fileMatches = [];

        if (p.includePaths) {

            if (typeof p.includePaths == "string") {
                // Arrayify the string
                includePaths = [params.includePaths];
            }else if (Array.isArray(p.includePaths)) {
                // Set this array to the includepaths
                includePaths = p.includePaths;
            }

            var includePath = '';

            for (var y = 0; y < includePaths.length; y++) {

                includePath = path.join(includePaths[y], matches[3].replace(/['"]/g, '')).trim();

                if (fs.existsSync(includePath)) {

                    var globResults = glob.sync(includePath, {mark: true});
                    fileMatches = fileMatches.concat(globResults);

                }

            }

        } else {

            var globResults = glob.sync(fullPath, {mark: true});
            fileMatches = fileMatches.concat(globResults);

        }

        try {

            fileMatches.forEach(function(value){

                var _inc = _parse(value);

                if (_inc.length > 0) {

                    var ind = (jsonVar.trim() == '*') ? indName(value) : jsonVar;
                    output[ind] = _inc;
                }
            })

        } catch (err) {
            console.log(err)
        }

    }
}

/**
 * Make Angular Template
 * @param p
 * @param json
 * @returns {string}
 */
function angularTemplate (p, json) {

    var prefix = (p.prefix !== "") ? p.prefix + "." : "",
        tpl = 'angular.module("'+ prefix +  p.filename +'",["ng"]).run(["$templateCache",';

    tpl += 'function($templateCache) {';

    for (var key in json) {
        if (json.hasOwnProperty(key)) {
            tpl += '$templateCache.put("'+ key +'",';
            tpl += JSON.stringify(json[key]);
            tpl += ');'
        }
    }

    tpl += '}])';

    return tpl;
}

/**
 * Module Exports htmlToJson
 * @param p
 * @returns {*}
 */
module.exports = function(p) {

    var sr = false;

    p = p || {};

    if (isEmpty(p.filename)) {
        sr = true;
    }

    function htmToJson(file, callback) {

        if (file.isNull()) {
            throw new plugError('gulp-html-to-json', 'File is Null');
        }

        if (file.isStream()) {
            throw new plugError('gulp-html-to-json', 'stream not supported');
        }

        if (file.isBuffer()) {

            var outputJson = {};

            htmlToJsonController(String(file.contents), file.path, outputJson, p);

            p.filename = indName(file.path, p);
            p.prefix || (p.prefix = "");
            p.useAsVariable || (p.useAsVariable = false);
            p.isAngularTemplate || (p.isAngularTemplate = false);

            if(p.isAngularTemplate) {
                var output = angularTemplate(p, outputJson);
                file.path = replaceFilename(file.path, p.filename, p.useAsVariable);
                file.contents = new Buffer(output);

            } else {
                file.path = replaceFilename(file.path, p.filename, p.useAsVariable);

                var exVars = (p.useAsVariable) ? "var " + p.filename + "=" : "";
                file.contents = new Buffer(exVars + JSON.stringify(outputJson));
            }
        }

        if (sr) {
            delete  p['filename'];
        }

        callback(null, file);
    }

    return es.map(htmToJson)
};
