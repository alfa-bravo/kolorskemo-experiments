# -*- coding: utf-8 -*-

# Uncomment this to force CPU computation
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from functools import reduce
from operator import xor
from scipy.spatial import KDTree
from keras.models import load_model
import json
import sys
import numpy as np


PALETTE_NUM_COLORS = 5
COLOR_NUM_COMPONENTS = 4

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
                min_nearest = reduce(lambda a, b: set(a) | set(b) if len(b) < min_number_of_categories else set(b), cs)
                # Re-vectorize pre-analyzed categories
                yield min_nearest, [1 if cat in min_nearest else 0 for cat in self._categories]
        stuff = list(wrap())
        nearest, encoded = list(zip(*stuff))
        return nearest, np.array(encoded)

def sorted_order(y):
    y_idx = list(enumerate(y))
    y_idx.sort(key=lambda e: e[1], reverse=True)
    y_ord, y_val = list(zip(*y_idx))
    return list(y_ord)

def prep_xy(vectorizer, data_):
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

def hash_list(list_):
    return reduce(lambda h, e: xor(h, hash(e)), list_, 0)

with open(sys.argv[1], 'r') as f:
    encoding_info = json.load(f)

categories = encoding_info['categories']
XY_encoded = encoding_info['XY']
XY = [(x, [categories[y] for y in Y]) for x, Y in XY_encoded]

encoder = KDEncoder(XY, categories)

model = load_model(sys.argv[2])

request = json.load(sys.stdin)

colors = request['colors']
n_colors = len(colors)
input_shape = (PALETTE_NUM_COLORS, len(categories))

# Process X
min_neighbors, X = encoder.transform(colors)
X = zero_padded_array(X, input_shape)
X = np.array([X])

y = model.predict(X)
y0 = y[0]
y0_n = y0 / max(y0)

top_indices = sorted_order(y0)
#top_categories = list(map(lambda i: categories[i], top_indices))

top_categories = {categories[i] for i in top_indices if y0_n[i] > 0.1}
predicted_name = ' '.join(top_categories)

output = {'predicted-name': predicted_name}

if 'DEBUG' in os.environ:
    import matplotlib.pyplot as plt
    plt.bar(x=categories, height=y0_n)
    plt.xticks(rotation=90)
    plt.show()
    print(min_neighbors)

print(json.dumps(output))