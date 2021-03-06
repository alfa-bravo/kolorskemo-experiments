# -*- coding: utf-8 -*-

# Uncomment this to force CPU computation
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
import sys
import numpy as np
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

stop_words = {'the', 'and', 'or', 'in', 'with', 'for', 'of', 'by', 'to' 'con', 'de', 'y', 'et', '&'}

COLOR_DATA_MAX_LENGTH = 5
RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)

def nn_model(n_colors, n_color_components, n_categories):
    N = 4 * n_categories
    model = Sequential([
        Dense(N, activation='relu', input_shape=(n_colors, n_color_components)),
        Flatten(),
        Dense(n_categories, kernel_initializer='normal', activation='softmax')
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

def prep_values(R, max_length):
    # Add alpha indicator variable and zero padding for input
    X = [[[1.0] + color for color in colors] + [[0.0, 0.0, 0.0, 0.0]] * (max_length - len(colors)) for colors in R]
    X = np.asarray(X)
    return X

def sorted_order(y):
    y_idx = list(enumerate(y))
    y_idx.sort(key=lambda e: e[1], reverse=True)
    y_ord, y_val = list(zip(*y_idx))
    return list(y_ord)

data_file_path = sys.argv[1]
with open(data_file_path, 'r', encoding='utf-8') as data_file:
    data = json.load(data_file)

data = dict(data)

s = list(data.keys())
R = list(data.values())

vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words)
vectorizer.fit(s)
categories = vectorizer.get_feature_names()

model = nn_model(COLOR_DATA_MAX_LENGTH, 4, len(categories))

s_train = None
X_test = None

for i in range(10):
    R_train, R_test, s_train, s_test = train_test_split(R, s, test_size=0.2, random_state=RANDOM_SEED)

    X_train = prep_values(R_train, COLOR_DATA_MAX_LENGTH)
    X_test = prep_values(R_test, COLOR_DATA_MAX_LENGTH)

    y_train = prep_keys(vectorizer, s_train)
    y_test = prep_keys(vectorizer, s_test)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=20, verbose=2)

for (name, x) in zip(s_test, X_test):
    x = np.asarray([x])
    y = model.predict(x)[0]
    y_top = sorted_order(y)[:3]
    top_labels = list(map(lambda i: categories[i], y_top))
    predicted_name = ' '.join(top_labels)
    print(name, "predicted as", predicted_name)

# if len(sys.argv) >= 3:
#     model.save(sys.argv[2])