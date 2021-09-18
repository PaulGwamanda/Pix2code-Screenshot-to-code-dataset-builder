from os import listdir
import numpy as np
from numpy import array
import keras
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential, model_from_json
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, GRU, TimeDistributed, RepeatVector, LSTM, concatenate, Input, Reshape, Dense
from keras.optimizers import RMSprop
print(keras.__version__)

dir_name = 'dataset/train/'
bootstrap_vocab = 'assets/bootstrap.vocab'

weights_path = 'weights/'
weights_hdf5 = "weights-epoch-{epoch:04d}--loss-{loss:.4f}.hdf5"

epochs = 1

# Read a file and return a string
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
image_model.add(Conv2D(16, (3,3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
image_model.add(Conv2D(32, (3,3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
image_model.add(Conv2D(64, (3,3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(128, (3,3), activation='relu', padding='same'))

image_model.add(Flatten())
image_model.add(Dense(1024, activation='relu'))
image_model.add(Dropout(0.3))
image_model.add(Dense(1024, activation='relu'))
image_model.add(Dropout(0.3))

image_model.add(RepeatVector(max_length))

visual_input = Input(shape=(256, 256, 3,))
encoded_image = image_model(visual_input)

language_input = Input(shape=(max_length,))
language_model = Embedding(vocab_size, 50, input_length=max_length, mask_zero=True)(language_input)
language_model = LSTM(128, return_sequences=True)(language_model)
language_model = LSTM(128, return_sequences=True)(language_model)

#Create the decoder
decoder = concatenate([encoded_image, language_model])
decoder = LSTM(512, return_sequences=True)(decoder)
decoder = LSTM(512, return_sequences=False)(decoder)
decoder = Dense(vocab_size, activation='softmax')(decoder)

# Compile the model
model = Model(inputs=[visual_input, language_input], outputs=decoder)
optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# serialize model to `
model_json = model.to_json()
with open(weights_path + 'model.json', "w") as json_file:
    json_file.write(model_json)
    print("Saved model to disk")

#Save the weights.hdf5 file
checkpoint = ModelCheckpoint(weights_hdf5, monitor='val_loss', verbose=1)
callbacks_list = [checkpoint]

steps = len(texts)
print('steps: ', steps)

# test the data generator
generator = data_generator(texts, train_features, 1, 150)
model.fit_generator(generator, steps_per_epoch=steps, epochs=epochs, callbacks=callbacks_list, verbose=1)

model.save(weights_path + weights_hdf5)