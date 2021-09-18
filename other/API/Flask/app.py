from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division, print_function

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from PIL import Image
from keras.models import load_model, model_from_json
import json

import numpy as np
import json
import string
import random
import codecs
import os
import glob

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define a flask app
app = Flask(__name__)

upload_folder = 'app/data/uploads/'
prediction_folder = 'app/data/prediction/'
conversion_folder = 'app/data/conversion/'

weights = "app/data/model/weights/650_3.hdf5"
model_json = 'app/data/model/_model.json'
bootstrap_vocab = 'app/data/model/bootstrap.vocab'
dsl_path = "app/data/model/web-dsl-mapping.json"

# Read a file and return a string
def load_doc(filename):
    file = codecs.open(filename, 'r', encoding='utf-8', errors='ignore')
    text = file.read()
    file.close()
    return text

@app.route('/', methods=['GET'])
def index():
    return render_template('/index.html', title='Home')

# @app.route('/prediction', methods=['GET'])
# def test():
#     return render_template('/prediction.html', title='Test')

# Section for creating new folders
def create_process_folder(create_folder):
    if not os.path.exists(create_folder):
        os.makedirs(create_folder)

def clean_process_folder(clean_folder):
    for f in clean_folder:
        os.remove(f)

create_process_folder(upload_folder)
create_process_folder(prediction_folder)
create_process_folder(conversion_folder)


@app.route('/upload', methods=['GET', 'POST'])
@cross_origin()
def upload():
    if request.method == 'POST':
        # Clearing session
        from keras import backend as K
        K.clear_session()

        # Empty contents of prediction and uploads folder
        upload_folder_files = glob.glob(upload_folder + '*')
        prediction_folder_files = glob.glob(prediction_folder + '*')
        conversion_folder_files = glob.glob(conversion_folder + '*')

        clean_process_folder(upload_folder_files)
        clean_process_folder(prediction_folder_files)
        clean_process_folder(conversion_folder_files)

        #Clean the upload folder
        print('Cleaning folder')

        def load_data():
            print("Uploading photo")
            # Get the file from post request
            file = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            uploaded_image = os.path.join(
                basepath, upload_folder, secure_filename(file.filename))
            file.save(uploaded_image)

            images = []

            im = Image.open(uploaded_image)
            im.save('app/data/conversion/predict.png')

            # Load the images already prepared in arrays
            if uploaded_image[-3:] == "npz":
                image = np.load(uploaded_image)
                images.append(image['features'])

            # Convert images to numpy arrays
            else:
                class Utils:
                    @staticmethod
                    def sparsify(label_vector, output_size):
                        sparse_vector = []

                        for label in label_vector:
                            sparse_label = np.zeros(output_size)
                            sparse_label[label] = 1

                            sparse_vector.append(sparse_label)

                        return np.array(sparse_vector)

                    @staticmethod
                    def get_preprocessed_img(img_path, image_size):
                        import cv2

                        # Crop image on the sides using edge detection here
                        import numpy as np
                        # img = cv2.imread("test.png")
                        img = cv2.imread(img_path)
                        # blurred = cv2.blur(img, (3, 3))
                        canny = cv2.Canny(img, 50, 200)

                        # find the non-zero min-max coords of canny
                        pts = np.argwhere(canny > 0)
                        y1, x1 = pts.min(axis=0)
                        y2, x2 = pts.max(axis=0)

                        ## crop the region
                        cropped = img[y1:y2, x1:x2]
                        cv2.imwrite(img_path, cropped)

                        # display
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, (image_size, image_size))
                        img = img.astype('float64')
                        img /= 255

                        return img

                    @staticmethod
                    def show(image):
                        import cv2
                        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
                        cv2.imshow("view", image)
                        cv2.waitKey(0)
                        cv2.destroyWindow("view")

                print("Converting images to numpy arrays...")

                input_path = conversion_folder
                output_path = prediction_folder
                IMAGE_SIZE = 256

                for f in os.listdir(input_path):
                    if f.find(".png") != -1:
                        img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)
                        file_name = f[:f.find(".png")]

                        np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
                        retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

                        assert np.array_equal(img, retrieve)

                    elif f.find(".jpg") != -1:
                        img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)
                        file_name = f[:f.find(".jpg")]

                        np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
                        retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

                        assert np.array_equal(img, retrieve)

                    elif f.find(".jpeg") != -1:
                        img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)
                        file_name = f[:f.find(".jpeg")]

                        np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
                        retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

                        assert np.array_equal(img, retrieve)

                for numpyImage in os.listdir(output_path):
                    image = np.load(output_path + numpyImage)
                    images.append(image['features'])

                print("Numpy arrays saved in {}".format(output_path))
            images = np.array(images, dtype=float)
            return images

        # Initialize the function to create the vocabulary
        tokenizer = Tokenizer(filters='', split=" ", lower=False)

        # Create the vocabulary
        tokenizer.fit_on_texts([load_doc(bootstrap_vocab)])
        train_features = load_data()

        # load model and weights
        json_file = open(model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        # loaded_model = load_model(weights)
        print("Loaded model from disk")

        # print("Loading model for prediction", model.summary())

        # map an integer to a word
        def word_for_id(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None

        # generate a description for an image
        def generate_desc(model, tokenizer, photo, max_length):
            photo = np.array(photo)
            # seed the generation process
            in_text = '<START> '
            # iterate over the whole length of the sequence
            # print('\nPrediction---->\n\n<START> ', end='')
            print('Compiling website')
            for i in range(750):
                # integer encode input sequence
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                # pad input
                sequence = pad_sequences([sequence], maxlen=max_length)
                # predict next word
                yhat = loaded_model.predict([photo, sequence], verbose=0)
                # convert probability to integer
                yhat = np.argmax(yhat)
                # map integer to word
                word = word_for_id(yhat, tokenizer)
                # stop if we cannot map the word
                if word is None:
                    break
                # append as input for generating the next word
                in_text += word + ' '
                # stop if we predict the end of the sequence
                print(word + ' ', end='')
                if word == '<END>':
                    break
            return in_text
        max_length = 48

        # evaluate the skill of the model
        def evaluate_model(model, photos, tokenizer, max_length):
            predicted = list()
            # step over the whole set
            yhat = generate_desc(model, tokenizer, photos, max_length)
            # store actual and predicted
            predicted.append(yhat.split())
            # calculate BLEU score
            # bleu = corpus_bleu(actual, predicted)
            return predicted

        predicted = evaluate_model(loaded_model, train_features, tokenizer, max_length)

        class Node:
            def __init__(self, key, parent_node, content_holder):
                self.key = key
                self.parent = parent_node
                self.children = []
                self.content_holder = content_holder

            def add_child(self, child):
                self.children.append(child)

            def show(self):
                for child in self.children:
                    child.show()

            def render(self, mapping, rendering_function=None):
                content = ""
                for child in self.children:
                    placeholder = child.render(mapping, rendering_function)
                    if placeholder is None:
                        self = None
                        return
                    else:
                        content += placeholder

                value = mapping.get(self.key, None)
                if value is None:
                    self = None
                    return None
                if rendering_function is not None:
                    value = rendering_function(self.key, value)

                if len(self.children) != 0:
                    value = value.replace(self.content_holder, content)

                return value

        class Utils:
            @staticmethod
            def get_random_text(length_text=10, space_number=1, with_upper_case=True):
                results = []
                while len(results) < length_text:
                    char = random.choice(string.ascii_letters[:26])
                    results.append(char)
                if with_upper_case:
                    results[0] = results[0].upper()

                current_spaces = []
                while len(current_spaces) < space_number:
                    space_pos = random.randint(2, length_text - 3)
                    if space_pos in current_spaces:
                        break
                    results[space_pos] = " "
                    if with_upper_case:
                        results[space_pos + 1] = results[space_pos - 1].upper()

                    current_spaces.append(space_pos)

                return ''.join(results)

        def render_content_with_text(key, value):
            if FILL_WITH_RANDOM_TEXT:
                if key.find("btn") != -1:
                    value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text())
                elif key.find("title") != -1:
                    value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=5, space_number=0))
                elif key.find("txt") != -1:
                    value = value.replace(TEXT_PLACE_HOLDER,
                                          Utils.get_random_text(length_text=56, space_number=7, with_upper_case=False))
            return value

        class Compiler:
            def __init__(self, dsl_mapping_file_path):
                with open(dsl_mapping_file_path) as data_file:
                    self.dsl_mapping = json.load(data_file)

                self.opening_tag = self.dsl_mapping["opening-tag"]
                self.closing_tag = self.dsl_mapping["closing-tag"]
                self.content_holder = self.opening_tag + self.closing_tag

                self.root = Node("body", None, self.content_holder)

            def compile(self, tokens, output_file_path):
                dsl_file = tokens

                # Parse fix
                dsl_file = dsl_file[1:-1]
                dsl_file = ' '.join(dsl_file)
                dsl_file = dsl_file.replace('{', '{8').replace('}', '8}8')
                dsl_file = dsl_file.replace(' ', '')
                dsl_file = dsl_file.split('8')
                dsl_file = list(filter(None, dsl_file))
                # End Parse fix

                current_parent = self.root

                for token in dsl_file:
                    token = token.replace(" ", "").replace("\n", "")
                    if token.find(self.opening_tag) != -1:
                        token = token.replace(self.opening_tag, "")

                        element = Node(token, current_parent, self.content_holder)
                        current_parent.add_child(element)
                        current_parent = element
                    elif token.find(self.closing_tag) != -1:
                        current_parent = current_parent.parent
                    else:
                        tokens = token.split(",")
                        for t in tokens:
                            element = Node(t, current_parent, self.content_holder)
                            current_parent.add_child(element)

                output_html = self.root.render(self.dsl_mapping, rendering_function=render_content_with_text)
                if output_html is None:

                    return "Parsing Error"

                with open(output_file_path, 'w') as output_file:
                    output_file.write(output_html)
                return output_html

        print("Compiling website")

        FILL_WITH_RANDOM_TEXT = True
        TEXT_PLACE_HOLDER = "[]"

        # Compile the tokens into HTML and css
        compiler = Compiler(dsl_path)

        # compiled_website = compiler.compile(predicted[0], "Editor/advanced/demo/offcanvas/index.html")
        compiled_website = compiler.compile(predicted[0], "testing.html")
        print(compiled_website)

        # K.clear_session()
        print("Clearing session")

        # Empty contents of prediction and uploads folder
        upload_folder_files = glob.glob(upload_folder + '*')
        prediction_folder_files = glob.glob(prediction_folder + '*')
        conversion_folder_files = glob.glob(conversion_folder + '*')

        for f in upload_folder_files:
            os.remove(f)
        for f in prediction_folder_files:
            os.remove(f)
        for f in conversion_folder_files:
            os.remove(f)

        return jsonify(compiled_website)

if __name__ == '__main__':
    app.run(threaded=False)
    app.run(debug=True)