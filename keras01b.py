# -*- coding: utf-8 -*-
import json
import sys
import numpy as np
from keras.layers import *
from keras.models import Model, Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

stop_words = {'the', 'and', 'or', 'in', 'with', 'for', 'of', 'by', 'to' 'con', 'de', 'y', 'et', '&'}

LABEL_MAX_CATEGORIES = 6
COLOR_DATA_MAX_LENGTH = 5
RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)

def nn_model(n_colors, n_color_components, n_categories, n_terms):
    '''
    Requirements
    -----
    - Inputs are color vectors concatenated with multi-one-hot encoded labels
    - Dynamic input shape to handle variable number of colors
    '''
    # a = 4 * n_categories
    # model = Sequential([
    #     Dense(a, activation='relu', input_shape=input_vector_shape), Dropout(0.1),
    #     Dense(a, activation='relu'), Dropout(0.1),
    #     Flatten(),
    #     Dense(n_categories, kernel_initializer='normal', activation='softmax')
    # ])

    inp = Input(shape=(n_colors, n_color_components))
    x = inp
    x = Dense(n_terms * n_categories, activation='relu')(x)
    x = Dense(n_terms * n_categories, activation='relu')(x)
    x = Dense(n_terms * n_categories, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(n_terms * n_categories, activation='softmax')(x)
    out = Reshape((n_terms, n_categories))(x)

    model = Model(inp, out)

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model

# def prep_keys(vectorizer, s):
#     y = vectorizer.transform(s).toarray()
#     # Frequency to multi-one-hot
#     limiter = np.vectorize(lambda t: 0 if t == 0 else 1)
#     X = limiter(y)
#     return y

def prep_keys2(vectorizer, encoder, s, vector_size):
    cat_labels = np.asarray(vectorizer.inverse_transform(vectorizer.transform(s)))
    cat_labels = [[[l] for l in labels] + [['']] * (vector_size - labels.shape[0]) for labels in cat_labels]
    vectors = [encoder.transform(l).toarray() for l in cat_labels]
    vectors = np.stack(vectors)
    return vectors

def prep_values(R, max_length):
    #R = [[f for color in colors for f in color] for colors in R]
    #length = len(max(R, key=len))
    # R = [[[f, 1.0] for f in fs] + [[0.0, 0.0]] * (max_length - len(fs)) for fs in R]
    # append indicator variable
    R = [[[1.0] + color for color in colors] + [[0.0, 0.0, 0.0, 0.0]] * (max_length - len(colors)) for colors in R]
    R = np.asarray(R)
    return R

data_file_path = sys.argv[1]
with open(data_file_path, 'r', encoding='utf-8') as data_file:
    data = json.load(data_file)

data = dict(data)

s = list(data.keys())
R = list(data.values())

# Convert input to multi-categorical one-hot encoding
vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words)
vectorizer.fit(s)
categories = vectorizer.get_feature_names()

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit([[label] for label in categories]) # Wrap labels in lists?

model = nn_model(COLOR_DATA_MAX_LENGTH, 4, len(categories), LABEL_MAX_CATEGORIES)

s_train = None
X_test = None

for i in range(20):
    R_train, R_test, s_train, s_test = train_test_split(R, s, test_size=0.333, random_state=RANDOM_SEED)

    X_train = prep_values(R_train, COLOR_DATA_MAX_LENGTH)
    X_test = prep_values(R_test, COLOR_DATA_MAX_LENGTH)

    y_train = prep_keys2(vectorizer, encoder, s_train, LABEL_MAX_CATEGORIES)
    y_test = prep_keys2(vectorizer, encoder, s_test, LABEL_MAX_CATEGORIES)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=50, verbose=2)

def sorted_order(y):
    y_idx = list(enumerate(y))
    y_idx.sort(key=lambda e: e[1])
    y_ord, y_val = list(zip(*y_idx))
    return list(y_ord)
'''
for (name, x) in zip(s_test, X_test):
    x = np.asarray([x])
    y = model.predict(x)[0]
    y_top = sorted_order(y)[:3]
    top_labels = map(lambda i: labels[i], y_top)
    predicted_name = ' '.join(top_labels)
    print(name, "predicted as", predicted_name)
'''