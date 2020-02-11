# -*- coding: utf-8 -*-
import json
import sys
import numpy as np
from keras.layers import *
from keras.models import Model, Sequential
from keras import optimizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from functools import reduce

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D

stop_words = {'the', 'and', 'or', 'in', 'with', 'for', 'of', 'by', 'to' 'con', 'de', 'y', 'et', '&'}

COLOR_RESOLUTION = 21
RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)


def get_distribution(vectors, resolution, spread=0.05):
    r = resolution - 1
    grid = np.mgrid[0:resolution, 0:resolution, 0:resolution].T.reshape(-1, 3)
    normalized_points = grid / r
    cov = np.identity(3) * spread / resolution

    def get_distribution_single(v):
        rv = multivariate_normal(v, cov)
        return rv.pdf(normalized_points)

    values = reduce(lambda s, c: s + get_distribution_single(c),
                    vectors,
                    np.zeros(normalized_points.shape[0]))

    output_shape = (resolution, resolution, resolution)
    a = np.zeros(output_shape)
    a[(*grid.T,)] = values
    return grid, a


def show_distribution(points, values):
    normalized_points = points / (np.max(points) - 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cdict = {
        'red': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        'green': [[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
        'blue': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        'alpha': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    }
    cmap = LinearSegmentedColormap('foo',segmentdata=cdict)
    cmap2 = ListedColormap([(0.0, 0.5, 0.0, 0.0), (0.0, 0.5, 0.0, 1.0)])
    ax.scatter(normalized_points[:, 0],
               normalized_points[:, 1],
               normalized_points[:, 2],
               c=values[(*points.T,)], cmap=cmap)
    ax.margins(0, 0, 0)
    plt.show()


def encode_input(R, resolution):
    distributions = []
    for r in R:
        points, distribution = get_distribution(r, resolution, 1.0)
        distribution /= np.max(distribution)
        distribution = distribution ** 2
        #show_distribution(points, distribution)
        distributions.append(distribution)
    # Wrap values in channel dimension
    return np.array(distributions)[..., None]


def encode_output(vectorizer, encoder, s):
    n_cats = len(vectorizer.get_feature_names())
    cat_labels = vectorizer.inverse_transform(vectorizer.transform(s))
    cat_labels = [[[l] for l in labels] for labels in cat_labels]
    vectors = np.array([reduce(np.add, encoder.transform(l).toarray(),
                               np.zeros(n_cats))
                        for l in cat_labels])
    return vectors


def nn_model(resolution, n_categories):
    input_shape = (resolution, resolution, resolution, 1)
    inp = Input(shape=input_shape)
    x = inp
    x = Conv3D(128, 3, dilation_rate=1)(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Conv3D(256, 3)(x)
    x = Flatten()(x)
    x = Dense(4 * n_categories, activation='relu')(x)
    x = Dropout(0.02)(x)
    x = Dense(4 * n_categories, activation='relu')(x)
    x = Dropout(0.02)(x)
    x = Dense(4 * n_categories, activation='relu')(x)
    x = Dropout(0.02)(x)
    x = Dense(3 * n_categories, activation='relu')(x)
    x = Dropout(0.02)(x)
    x = Dense(3 * n_categories, activation='relu')(x)
    x = Dropout(0.02)(x)
    x = Dense(2 * n_categories, activation='relu')(x)
    x = Dropout(0.02)(x)
    x = Dense(n_categories, activation='relu')(x)
    x = BatchNormalization()(x)
    out = Dense(n_categories, activation='softmax')(x)
    model_ = Model(inp, out)
    opt = optimizers.SGD()
    model_.compile(loss='kullback_leibler_divergence',
                   optimizer=opt,
                   metrics=['accuracy'])
    return model_


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
# Encoder needs to work on 2D array
encoder.fit([[label] for label in categories])

model = nn_model(COLOR_RESOLUTION, len(categories))

s_train = None
X_test = None

print(model.summary())

R_train, R_test, s_train, s_test = train_test_split(R, s, test_size=0.333, random_state=RANDOM_SEED)

X_train = encode_input(R_train, COLOR_RESOLUTION)
X_test = encode_input(R_test, COLOR_RESOLUTION)

y_train = encode_output(vectorizer, encoder, s_train)
y_test = encode_output(vectorizer, encoder, s_test)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=50, verbose=2)


def sorted_order(y):
    y_idx = list(enumerate(y))
    y_idx.sort(key=lambda e: e[1], reverse=True)
    y_ord, y_val = list(zip(*y_idx))
    return list(y_ord)


for (name, x) in zip(s_test, X_test):
    x = np.asarray([x])
    y = model.predict(x)[0]
    y_top = sorted_order(y)[:3]
    top_labels = map(lambda i: categories[i], y_top)
    predicted_name = ' '.join(top_labels)
    print(name, "predicted as", predicted_name)
