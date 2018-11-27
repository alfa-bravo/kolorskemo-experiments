# -*- coding: utf-8 -*-

import json
import sys
from functools import reduce
from operator import xor

import numpy as np
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from copy import copy, deepcopy

stop_words = {'the', 'and', 'or', 'in', 'with', 'for', 'of', 'by', 'to' 'con', 'de', 'y', 'et', '&', 'after', 'before'}

PALETTE_NUM_COLORS = 5
COLOR_NUM_COMPONENTS = 4
RANDOM_SEED = 7

np.random.seed(RANDOM_SEED)

def nn_model(n_categories):
    A = 8 * n_categories
    model = Sequential([
        Flatten(),
        Dense(A, activation='sigmoid'),
        Dense(A, activation='sigmoid'),
        Dense(A, activation='sigmoid'),
        Dense(A, activation='sigmoid'),
        Dense(n_categories, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def prep_keys(vectorizer, s):
    # Frequency to multi-category
    # Convert input to multi-categorical one-hot encoding
    y = vectorizer.transform(s).toarray()
    limiter = np.vectorize(lambda t: 0 if t == 0 else 1)
    y = limiter(y)
    return y

def hash_list(list_):
    return reduce(lambda h, e: xor(h, hash(e)), list_, 0)


class KDMap():
    def lookup_query(self, x, *args, **kwargs):
        distances, indices = self._kdt.query(x, *args, **kwargs)
        return distances, [[self._lut[hash_list(self._kdt.data[i])] for i in idx] for idx in indices]

    def get_kdt(self):
        return deepcopy(self._kdt)

    def __init__(self, XY):
        Xs, Ys = zip(*XY)
        Xs = np.array(Xs)
        self._kdt = KDTree(Xs)
        self._lut = { hash_list(x): y for x, y in XY}

class KDEncoder():
    def __init__(self, XY, categories):
        self._kdm = KDMap(XY)
        self._categories = categories
    def transform(self, X, min_number_of_categories=3):
        distances, category_sets = self._kdm.lookup_query(X, k=min_number_of_categories)
        def wrap():
            for cs in category_sets:
                min_nearest = reduce(lambda a, b: a | b if len(b) < min_number_of_categories else b, cs)
                # Re-vectorize pre-analyzed categories
                yield min_nearest, [1 if cat in min_nearest else 0 for cat in self._categories]
        stuff = list(wrap())
        nearest, encoded = list(zip(*stuff))
        return nearest, np.array(encoded)

def prep_xy(vectorizer, data_, encoding=None):
    def catvec1(x):
        return vectorizer.inverse_transform(vectorizer.transform([x]))[0].tolist()
    flat = [(c, hash_list(c), catvec1(s)) for s, r in data_.items() for c in r]
    orig = {k: c for c, k, _ in flat}
    lookup = {k: set() for k in orig.keys()}
    for c, k, cv in flat:
        lookup[k] |= set(cv)
    return [(orig[k], cv) for k, cv in lookup.items()]


def zero_padded_array(array, shape):
    padded = np.zeros(shape)
    padded[:array.shape[0],:array.shape[1]] = array
    return padded

data_file_path = sys.argv[1]
output_files_prefix = sys.argv[2]


with open(data_file_path, 'r', encoding='utf-8') as data_file:
    data = json.load(data_file)

data = dict(data)

s = list(data.keys())
R = list(data.values())

vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words)
vectorizer.fit(data.keys())
categories = vectorizer.get_feature_names()
XY = prep_xy(vectorizer, data)
input_shape = (PALETTE_NUM_COLORS, len(categories))

encoder = KDEncoder(XY, categories)

model = nn_model(len(categories))


for i in range(10):
    R_train, R_test, s_train, s_test = train_test_split(R, s, test_size=0.3, random_state=RANDOM_SEED)

    _, X_train = zip(*[encoder.transform(r) for r in R_train])
    _, X_test = zip(*[encoder.transform(r) for r in R_test])


    X_train = np.array([zero_padded_array(x, input_shape) for x in X_train])
    X_test = np.array([zero_padded_array(x, input_shape) for x in X_test])

    y_train = prep_keys(vectorizer, s_train)
    y_test = prep_keys(vectorizer, s_test)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=100, verbose=2)

if len(sys.argv) >= 3:
    XY_encoded = [(x, [categories.index(y) for y in Y]) for x, Y in XY]

    encoding_info = {
        "categories": categories,
        "XY": XY_encoded
    }

    model.save(f'{output_files_prefix}.h5')
    with open(f'{output_files_prefix}-encoding.json', 'w') as f:
        json.dump(encoding_info, f)