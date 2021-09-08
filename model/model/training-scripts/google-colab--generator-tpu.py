import os
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential, model_from_json
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, TimeDistributed, RepeatVector, LSTM, concatenate , Input, Reshape, Dense
import numpy as np
import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

from google.colab import drive
drive.mount('/content/gdrive')

# !ls "/content/gdrive/My Drive/weights"
mydrive = "/content/gdrive/My Drive"
dir_name = mydrive+"/data/train/dataset/"
weights = mydrive+"/weights/org-weights-epoch-0265--loss-0.193.hdf5"
bootstrap_vocab = mydrive+"/weights/bootstrap.vocab"
model_json = mydrive+"/weights/model.json"

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_data(data_dir):
    text = []
    images = []
    # Load all the files and order them
    all_filenames = listdir(data_dir)
    all_filenames.sort()
    for filename in (all_filenames):
        if filename[-3:] == "npz":
            # Load the images already prepared in arrays
            image = np.load(data_dir+filename)
            images.append(image['features'])
        else:
            # Load the boostrap tokens and rap them in a start and end tag
            syntax = '<START> ' + load_doc(data_dir+filename) + ' <END>'
            # Seperate all the words with a single space
            syntax = ' '.join(syntax.split())
            # Add a space after each comma
            syntax = syntax.replace(',', ' ,')
            text.append(syntax)
    images = np.array(images, dtype=float)
    return images, text


train_features, texts = load_data(dir_name)

# Initialize the function to create the vocabulary
tokenizer = Tokenizer(filters='', split=" ", lower=False)

# Create the vocabulary
tokenizer.fit_on_texts([load_doc(bootstrap_vocab)])

# Add one spot for the empty word in the vocabulary
vocab_size = len(tokenizer.word_index) + 1

max_length = 48

def create_sequences(texts, features, max_sequence):
    X, y, image_data = list(), list(), list()
    sequences = tokenizer.texts_to_sequences(texts)
    for img_no, seq in enumerate(sequences):
        for i in range(1, len(seq)):
            # Add the sentence until the current count(i) and add the current count to the output
            in_seq, out_seq = seq[:i], seq[i]
            # Pad all the input token sentences to max_sequence
            in_seq = pad_sequences([in_seq], maxlen=max_sequence)[0]
            # Turn the output into one-hot encoding
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # Add the corresponding image to the boostrap token file
            image_data.append(features[img_no])
            # Cap the input sentence to 48 tokens and add it
            X.append(in_seq[-48:])
            y.append(out_seq)
    return np.array(image_data), np.array(X), np.array(y)


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, features, n_step, max_sequence):
    # loop until we finish training
    while 1:
        # loop over photo identifiers in the dataset
        for i in range(0, len(descriptions)):
            Ximages, XSeq, y = list(), list(), list()
            for j in range(i, min(len(descriptions), i+n_step)):
                image = features[j]
                # retrieve text input
                desc = descriptions[j]
                # generate input-output pairs
                in_img, in_seq, out_word = create_sequences([desc], [image], max_sequence)
                for k in range(len(in_img)):
                    Ximages.append(in_img[k])
                    XSeq.append(in_seq[k])
                    y.append(out_word[k])
                    # yield this batch of samples to the model
                yield ([array(Ximages), array(XSeq)], array(y))

#Create the encoder
image_model = Sequential()
image_model.add(Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3,)))
image_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=8))
image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=8))
image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=8))
image_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

image_model.add(Flatten())
image_model.add(Dense(256, activation='relu'))
image_model.add(Dropout(0.3))
image_model.add(Dense(256, activation='relu'))
image_model.add(Dropout(0.3))

image_model.add(RepeatVector(max_length))

visual_input = Input(shape=(256, 256, 3,))
encoded_image = image_model(visual_input)

language_input = Input(shape=(max_length,))
language_model = Embedding(vocab_size, 50, input_length=max_length, mask_zero=True)(language_input)
language_model = LSTM(128, return_sequences=True)(language_model)
language_model = LSTM(128, return_sequences=True)(language_model)

# Create the decoder
decoder = concatenate([encoded_image, language_model])
decoder = LSTM(256, return_sequences=True)(decoder)
decoder = LSTM(256, return_sequences=False)(decoder)
decoder = Dense(vocab_size, activation='softmax')(decoder)

# Compile the model
# model = Model(inputs=[visual_input, language_input], outputs=decoder)
# optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# load json from existing model to continue training
json_file = open(model_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
print("Loaded model from disk")

# Compile the model
optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.load_weights(weights)
print("Loaded weights from disk")

#Save the model for every 2nd epoch
filepath= "weights-epoch-{epoch:04d}-loss-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_weights_only=True, period=1)
callbacks_list = [checkpoint]

# test the data generator
generator = data_generator(texts, train_features, 1, 150)
model.fit_generator(generator, steps_per_epoch=2500, epochs=1, callbacks=callbacks_list, verbose=1)
model.save(mydrive + '/output/weights-epoch-{epoch:04d}-loss-{loss:.4f}.hdf5')