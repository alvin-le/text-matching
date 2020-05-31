# !/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
from keras.preprocessing.text import Tokenizer
from operator import itemgetter
import tensorflow as tf
import numpy as np

W2V_EMB_FILE = './data/word_vector.txt'

def get_timestamp():
    now = int(round(time.time() * 1000))
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now / 1000))

def get_coefs(word, *arr):
    if 1 == len(arr):
        return '[UNK]',np.zeros(100)
    else:
        vector = arr[:100]
    return word, np.asarray(vector, dtype='float32')

def get_emb_matrix(data_dir, max_features):
    data_file = os.path.join(data_dir, "data.txt")
    if not os.path.isfile(data_file) or not os.path.getsize(data_file):
        raise ValueError("data file [%s] not ready"  % (data_file))
    
    tokens = []
    with open(data_file) as inf:
        for line in inf:
            cols = line.strip().split("\t")
            if 2 > len(cols):
                continue
            tokens.extend(cols[1].strip().split(" "))
    
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(tokens)
    word_index = tokenizer.word_index
    dict_sorted = sorted(word_index.iteritems(), key=itemgetter(1), reverse=False)        
    
    output_file = os.path.join(data_dir, "vocab.txt")
    with tf.gfile.GFile(output_file, "w") as writer:
        for k,v in dict_sorted:
            writer.write("%s\n" % (k))

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(W2V_EMB_FILE,'r'))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
