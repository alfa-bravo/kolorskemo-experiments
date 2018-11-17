# -*- coding: utf-8 -*-

import os, sys, json
from itertools import combinations
from itertools import chain
from scipy.spatial import KDTree
from operator import xor
from functools import reduce
from struct import pack, unpack

stop_words = ['the', 'and', 'or', 'in', 'with', 'for',
              'of', 'con', 'de', 'y', 'et', '&']

request = json.load(sys.stdin)
request_colors = request['colors']

data_file_path = sys.argv[1]
with open(data_file_path, 'r', encoding='utf-8') as data_file:
    data = json.load(data_file)

items = list(data.items())

def all_combinations(list_):
    for n in range(len(list_)):
        for e in combinations(list_, n + 1):
            yield e

words_to_colors = {}
colors_to_words = {}
hash_to_color = {}
all_colors = []

def ftoi(f):
    return unpack('I',pack('f', f))[0]

def hash_list(list_):
    return reduce(lambda h, e: xor(h, hash(e)), list_, 0)

for k, v in items:
    k_list = k.lower().split()
    #k_combos = list(map(lambda e: ' '.join(e), all_combinations(k_list)))
    #print(k_list)
    for word in k_list:
        if word in stop_words:
            continue
        if word not in words_to_colors:
            words_to_colors[word] = []
        for color in v:
            color_hash = hash_list(color)
            hash_to_color[color_hash] = color
            if color_hash not in colors_to_words:
                colors_to_words[color_hash] = []
                all_colors.append(color)
            if word not in colors_to_words[color_hash]:
                colors_to_words[color_hash].append(word)
            if color not in words_to_colors[word]:
                words_to_colors[word].append(color)

#for hash_, words in colors_to_words.items():
#    color = hash_to_color[hash_]
#    print(color, words)

tree = KDTree(all_colors)

distinct_words = set()
for color in request_colors:
    q1 = tree.query_ball_point(color, 0.1)[:3]
    result_colors = [all_colors[i] for i in q1]
    words = [colors_to_words[hash_list(color)] for color in result_colors]
    flat_words = chain.from_iterable(words)
    distinct_words.update(flat_words)
       
print(list(distinct_words))