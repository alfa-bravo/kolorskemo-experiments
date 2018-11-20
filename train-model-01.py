# -*- coding: utf-8 -*-

import json
import sys
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

stop_words = {'the', 'and', 'or', 'in', 'with', 'for', 'of', 'by', 'to' 'con', 'de', 'y', 'et', '&', 'after', 'before'}

PALETTE_NUM_COLORS = 5
COLOR_NUM_COMPONENTS = 4
RANDOM_SEED = 7

np.random.seed(RANDOM_SEED)

def nn_model(n_colors, n_color_components, n_categories):
    n = n_colors * n_color_components
    A = 8 * n_categories
    model = Sequential([
        Flatten(),
        Dense(A, activation='relu'),
        Dropout(0.1),
        Dense(A, activation='relu'),
        Dropout(0.5),
        Dense(A, activation='relu'),
        Dropout(1.0),
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
output_files_prefix = sys.argv[2]


with open(data_file_path, 'r', encoding='utf-8') as data_file:
    data = json.load(data_file)

data = dict(data)

s = list(data.keys())
R = list(data.values())

vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words)
vectorizer.fit(s)
categories = vectorizer.get_feature_names()

model = nn_model(PALETTE_NUM_COLORS, COLOR_NUM_COMPONENTS, len(categories))

s_train = None
X_test = None

for i in range(40):
    R_train, R_test, s_train, s_test = train_test_split(R, s, test_size=0.3, random_state=RANDOM_SEED)

    X_train = prep_values(R_train, PALETTE_NUM_COLORS)
    X_test = prep_values(R_test, PALETTE_NUM_COLORS)

    y_train = prep_keys(vectorizer, s_train)
    y_test = prep_keys(vectorizer, s_test)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=200, verbose=2)

if len(sys.argv) >= 3:
    model.save(f'{output_files_prefix}.h5')
    with open(f'{output_files_prefix}-categories.json', 'w') as f:
        json.dump(categories, f)