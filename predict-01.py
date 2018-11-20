# -*- coding: utf-8 -*-

# Uncomment this to force CPU computation
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
import sys
import numpy as np
from keras.models import load_model

PALETTE_NUM_COLORS = 5
COLOR_NUM_COMPONENTS = 4

def sorted_order(y):
    y_idx = list(enumerate(y))
    y_idx.sort(key=lambda e: e[1], reverse=True)
    y_ord, y_val = list(zip(*y_idx))
    return list(y_ord)

def prep_values(R, max_length):
    # Add alpha indicator variable and zero padding for input
    X = [[[1.0] + color for color in colors] + [[0.0, 0.0, 0.0, 0.0]] * (max_length - len(colors)) for colors in R]
    X = np.asarray(X)
    return X

with open(sys.argv[1], 'r') as f:
    categories = json.load(f)

model = load_model(sys.argv[2])

request = json.load(sys.stdin)

colors = request['colors']
n_colors = len(colors)

X = prep_values([colors], PALETTE_NUM_COLORS)

y = model.predict(X)
y0 = y[0]
top_indices = sorted_order(y0)[:n_colors]
top_categories = list(map(lambda i: categories[i], top_indices))
predicted_name = ' '.join(top_categories)

output = {'predicted-name': predicted_name}

print(json.dumps(output))