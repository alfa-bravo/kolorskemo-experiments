# -*- coding: utf-8 -*-

import sys, json
from itertools import combinations, chain, repeat
from scipy.spatial import KDTree
from operator import xor
from functools import reduce
from collections import Counter

stop_words = ['the', 'and', 'or', 'in', 'with', 'for',
              'of', 'con', 'de', 'y', 'et', '&', 'by', 'n', 'after']

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

def dist(a, b):
    '''
    :param a: vector list a
    :param b: vector list b
    :return: ||ab|| = sqrt((b - a)^2)
    '''
    ab = zip(a, b)
    return reduce(lambda sum, comp: sum + (comp[1] - comp[0])**2, ab, 0.0)**0.5

def proximity(a, b):
    return 1.0 / (1.0 + dist(a, b))

words_to_colors = {}
colors_to_words = {}
hash_to_color = {}
all_colors = []

def hash_list(list_):
    return reduce(lambda h, e: xor(h, hash(e)), list_, 0)

for k, v in items:
    k_list = k.lower().split()
    #k_combos = list(map(lambda e: ' '.join(e), all_combinations(k_list)))
    #print(k_list)
    for word in k_list:
        # Skip stop words
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


tree = KDTree(all_colors)

def query_color(color, radius):
    q1 = tree.query_ball_point(color, radius)
    if len(q1) == 0:
        q1 = [tree.query(color)[1]]
    result_colors = [all_colors[i] for i in q1]
    return result_colors, [colors_to_words[hash_list(c)] for c in result_colors]
    

def get_words_near_color(color, radius):
    result_colors, word_lists = query_color(color, radius)
    proximal_colors = [(c, proximity(color, c)) for c in result_colors]
    # Expand color1, proximity1, [word1, word2] to
    # word1, color1, proximity1
    # word2, color1, proximity1
    weighted_words = list(chain.from_iterable([zip(word_lists[i], repeat(c), repeat(p)) for i, (c, p) in enumerate(proximal_colors)]))
    flat_words = list(chain.from_iterable(word_lists))
    word_weights = {}
    for word, c, p in weighted_words:
        # Cumulative method weighs frequent colors more
        word_weights[word] = 1.0 - (1.0 - p) * (1.0 - word_weights[word]) if word in word_weights else p
    distinct_words = list(set(flat_words))
    distinct_words.sort(key=lambda word: (-word_weights[word], word))
    return [(word, word_weights[word]) for word in distinct_words]

for color in request_colors:
    word, *_ = get_words_near_color(color, 0.08)
    if word[1] > 0.9:
        print(word[0])