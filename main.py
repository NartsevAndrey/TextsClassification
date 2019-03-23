import numpy as np
import data_helpers
from w2v import train_word2vec
import keras
import datetime
from keras.engine.input_layer import Input, InputLayer
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
import os

N = 5

embedding_dim = 300
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

batch_size = 64
num_epochs = 1

sequence_length = 400
max_words = 5000

slash = "/"


def load_data():
    x, y, vocabulary, vocabulary_inverse = data_helpers.load_data()
    print(x.shape)
    print(y.shape)
    shuffle = np.random.permutation(np.arange(len(y)))
    print(shuffle.shape)
    x = x[shuffle]
    y = y[shuffle]
    train_len = int(len(x) * 0.8)
    x_train, y_train = x[:train_len], y[:train_len]
    x_valid, y_valid = x[train_len:], y[train_len:]
    return x_train, y_train, x_valid, y_valid, vocabulary_inverse, vocabulary


def prepare_test_sample(vocabulary):
    res = []
    s = data_helpers.preparation_sample_file()
    for sent in s.split("\n"):
        tmp = []
        for word in sent.split():
            if word in vocabulary:
                tmp.append(word)
								if len(tmp) <= sequence_length
								    res.append(tmp)
    gen = data_helpers.ExtendSent(res, size=sequence_length)
    return np.array([[vocabulary[word] for word in sentence] for sentence in gen])


def make_configuration():
    input_shape = (sequence_length,)
    model_input = Input(shape=input_shape)
    z = BatchNormalization()(model_input)
    z = Embedding(len(vocabulary_inv),
                  embedding_dim,
                  input_length=sequence_length,
                  name="embedding")(model_input)
    z = Dropout(dropout_prob[0])(z)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = BatchNormalization()(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = BatchNormalization()(z)
    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    z = BatchNormalization()(z)
    model_output = Dense(N, activation="softmax")(z)
    model = Model(model_input, model_output)
    return model


def initialize_weights(model, embedding_weights):
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])


def create_model():
    final_paths = "files_path.txt"
    model_path = "models" + slash + "model_5E.model"
    embedding_weights = train_word2vec(vocabulary_inv,
                                       final_paths=final_paths,
                                       model_paths=model_path,
                                       option="load")
																																							
    model = make_configuration()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    initialize_weights(model, embedding_weights)

    model.save("trained_models" + slash + "model_one")
    model.save_weights("trained_models" + slash + "model_one_weight")

    return model

				
with open("last_name.txt", "r", encoding="cp1251") as r:
    name_of_last_model = r.read().strip()

				
def loading_model():
    return load_model("trained_models" + slash + name_of_last_model)


def train_model():
    model = loading_model()
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        validation_data=(x_test, y_test),
                        verbose=1)
    new_name_model = "model_one" + str(datetime.datetime.now())[:-7]
    name_of_last_model = new_name_model
				
    with open("last_name.txt", "w", encoding="cp1251") as w:
        w.write(name_of_last_model)

    return model, name_of_last_model


def make_prediction(model, vocabulary):
    data = prepare_test_sample(vocabulary)
    predict = model.predict(data)
    prob = predict.sum(axis=0)
    norm = prob.sum()
    prob = [x / norm for x in prob]
    print(prob)


print("Load data...")
x_train, y_train, x_test, y_test, vocabulary_inv, vocabulary = load_data()
if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

#model = create_model()
model = loading_model()

#for i in range(2):
#    model, name_of_last_model = train_model()
#    print(name_of_last_model)
#    model.save("trained_models" + slash + name_of_last_model)
#    model.save_weights("trained_models" + slash + name_of_last_model + "_weight")
make_prediction(model, vocabulary)