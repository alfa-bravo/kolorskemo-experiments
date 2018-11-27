# -*- coding: utf-8 -*-

import json
import sys
from functools import reduce
from operator import xor

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KDTree
from copy import copy, deepcopy

stop_words = {'the', 'and', 'or', 'in', 'with', 'for', 'of', 'by', 'to' 'con', 'de', 'y', 'et', '&', 'after', 'before'}

PALETTE_NUM_COLORS = 5
COLOR_NUM_COMPONENTS = 4
RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)

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
                min_nearest = reduce(lambda a, b: a | b if len(a) < min_number_of_categories else a, cs)
                # Re-vectorize pre-analyzed categories
                yield min_nearest, [1 if cat in min_nearest else 0 for cat in self._categories]
        nearest, encoded = list(zip(*wrap()))
        return nearest, np.array(encoded)

def prep_xy(vectorizer, data_):
    def catvec1(x):
        return vectorizer.inverse_transform(vectorizer.transform([x]))[0].tolist()
    flat = [(c, hash_list(c), catvec1(s)) for s, r in data.items() for c in r]
    orig = {k: c for c, k, _ in flat}
    lookup = {k: set() for k in orig.keys()}
    for c, k, cv in flat:
        lookup[k] |= set(cv)
    return [(orig[k], cv) for k, cv in lookup.items()]

data_file_path = sys.argv[1]

with open(data_file_path, 'r', encoding='utf-8') as data_file:
    data = json.load(data_file)

data = dict(data)
vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words)
vectorizer.fit(data.keys())
categories = vectorizer.get_feature_names()
XY = prep_xy(vectorizer, data)

encoder = KDEncoder(XY, categories)

min_nearest, encoded = encoder.transform([[0.1, 0.2, 0.3], [0.2, 0.3, 0.5]])

print(encoded)