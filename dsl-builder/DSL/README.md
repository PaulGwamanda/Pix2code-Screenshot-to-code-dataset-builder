## Usage

> Run gulp:
>
```
$ gulp 
```

> This will generate the web-dsl-mapping.json, which is using the tokens and template reference from the /tokens folder:

```
{
  "header":"<header> <ul> <li>Nav 1</li> <li>Nav 2</li> <li>Nav 3</li> <li>Nav 4</li> </ul> </header>",
  "body":"<div class=\"container\"> <div class=\"wrapper\"> <h1>THIS IS THE BODY</h1> <p>This is underscore template tags <%= variable %></p> </div> </div>",
  "footer":"<footer> <div class=\"title\"> <h1>THIS IS THE FOOTER</h1> </div> </footer>"
}
```

## Options

In the file where you want want to compile you html, add a comment similar to this:

```javascript
//= key.name : relative/path/to/file.html
```

where key.name is the name want to associate with the html content in your json object.

If you use * as your key name like this :

```javascript
//= * : relative/path/to/file.html
```

It will automatically use the filename of your html as its key name.

If you want to use glob similar to commonly used in GruntJS, you may also to that like this:

```javascript
//= * : relative/path/to/**/*.html
```

Suggested key name is * so the it will use the filename as the keyname.

First sample code output:

```json
{
    "key.name" : "<div>your html content</div>"
}
```

Second sample code output:

```json
{
    "file" : "<div>your html content</div>"
}
```

Third sample will look into all html content inside the directory and output it like this:

```json
{
    "file" : "<div>your html content</div>",
    "file2" : "<div>your html content 2</div>"
}
```

```
Note: Code has been modified for my purposes from version of this plugin which can be found here: https://github.com/johnturingan/gulp-html-to-json
```