# -*- coding: utf-8 -*-

import json
import sys
import numpy as np
from keras.layers import Dense, Dropout, Add, BatchNormalization, Activation
from keras.models import Model, Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

stop_words = {'the', 'and', 'or', 'in', 'with', 'for', 'of', 'by', 'to' 'con', 'de', 'y', 'et', '&', 'after', 'before'}

COLOR_NUM_COMPONENTS = 3
RANDOM_SEED = 7

np.random.seed(RANDOM_SEED)


def color_model(n_components, n_categories):
    A = n_categories
    model = Sequential([
        Dense(A),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.1),
        Dense(A),
        BatchNormalization(),
        Dense(n_categories, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.build(input_shape=(None, n_components,))
    return model


def palette_model(n_categories):
    A = 2 * n_categories
    model = Sequential([
        Dense(A, activation='relu'),
        Dropout(0.1),
        Dense(A, activation='relu'),
        Dropout(0.1),
        Dense(A, activation='relu'),
        Dropout(0.1),
        Dense(n_categories, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.build(input_shape=(None, n_categories,))
    return model


def encode_keys(vectorizer, s):
    # Frequency to multi-category
    # Convert input to categorical stochastic encoding
    y = vectorizer.transform(s).toarray()
    #limiter = np.vectorize(lambda t: 0 if t == 0 else 1)
    #y = limiter(y)
    return y


def sorted_order(y):
    y_idx = list(enumerate(y))
    y_idx.sort(key=lambda e: e[1], reverse=True)
    y_ord, y_val = list(zip(*y_idx))
    return list(y_ord)


def expand_values(s_, R_):
    for y, X in zip(s_, R_):
        for v in X:
            yield y, v



data_file_path = sys.argv[1]
output_files_prefix = sys.argv[2]


with open(data_file_path, 'r', encoding='utf-8') as data_file:
    data = json.load(data_file)

data = dict(data)

# Color palette training
s = list(data.keys())
R = list(data.values())

# Single color training
q, P = zip(*list(expand_values(s, R)))


vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words)
vectorizer.fit(s)
categories = vectorizer.get_feature_names()

s_train = None
X_test = None



color_m = color_model(COLOR_NUM_COMPONENTS, len(categories))
palette_m = palette_model(len(categories))

# Remove palette model input layer
#palette_m_input = palette_m.layers.pop(0)


#for i in range(40):
#     R_train, R_test, s_train, s_test = train_test_split(
#         R, s, test_size=0.3, random_state=RANDOM_SEED)
#
#     # # Train models together (oops doesn't work)
#     # for palette, categories in zip(s_train, R_train):
#     #     color_ins = [color_m.input for _ in range(len(palette))]
#     #     color_outs = [color_m.output for _ in range(len(palette))]
#     #
#     #     add_layer = Add()(color_outs)
#     #     palette_out = palette_m(add_layer)
#     #     training_model = Model(color_ins, palette_out)
#     #     print(training_model.summary())
#     #     training_model.train_on_batch([categories], [palette])
#
#
#     #color_m.fit(P_train, q_train, validation_data=(P_test, q_test),
#     #            epochs=40, batch_size=200, verbose=2)


for i in range(10):
    P_train, P_test, q_train, q_test = train_test_split(
        P, q, test_size=0.3, random_state=RANDOM_SEED)

    y_train = encode_keys(vectorizer, q_train)
    y_test = encode_keys(vectorizer, q_test)

    X_train = np.array(P_train)
    X_test = np.array(P_test)

    color_m.fit(X_train, y_train, validation_data=(X_test, y_test),
                epochs=10, batch_size=100, verbose=2)


for i in range(40):
    R_train, R_test, s_train, s_test = train_test_split(R, s, test_size=0.3, random_state=RANDOM_SEED)

    y_train = encode_keys(vectorizer, s_train)
    y_test = encode_keys(vectorizer, s_test)

    X_train = np.array([np.sum(color_m.predict(np.array(palette)), axis=0) for palette in R_train])
    X_test = np.array([np.sum(color_m.predict(np.array(palette)), axis=0) for palette in R_test])

    print(X_train.shape)

    palette_m.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=200, verbose=2)

# if len(sys.argv) >= 3:
#     palette_m.save(f'{output_files_prefix}.h5')
#     with open(f'{output_files_prefix}-categories.json', 'w') as f:
#         json.dump(categories, f)