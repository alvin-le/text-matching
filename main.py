#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session

from model_builder import ModelBuilder
from text_matcher import TextMatcher
from cyclic_lr import CyclicLR
import utils as utils


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .tsv files.")

flags.DEFINE_string("output_dir", None, "The dir saved model weight.")

flags.DEFINE_string("model_name", None, "The name of the model.")

flags.DEFINE_float("min_lr", 0.001, "The min learning rate.")

flags.DEFINE_float("max_lr", 0.003, "The min learning rate.")

flags.DEFINE_integer("epochs", 5, "The training epochs.")

flags.DEFINE_integer("batch_size", 32, "The training batch size.")

flags.DEFINE_string("vocab_file", "data/vocab.txt", "The vocab file path.")

flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

flags.DEFINE_integer("max_seq_len", 21, "The max num of tokens for each query.")

flags.DEFINE_integer("input_dim", 111857, "The dim of input fea.")

flags.DEFINE_integer("max_features", 50000, "The num of max fea.")

flags.DEFINE_integer("units", 150, "The uints of hidden layer.")

flags.DEFINE_integer("num_filter", 128, "[optional] The num of filters if use GRU.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("is_fixed_emb", False, "Whether to fix input embedding matrix.")

def main(_):
    tf.gfile.MakeDirs(FLAGS.output_dir)

    if FLAGS.is_fixed_emb:
        emb_matrix = utils.get_emb_matrix(FLAGS.data_dir, FLAGS.max_features)
    
    clr = CyclicLR(base_lr=FLAGS.min_lr, max_lr=FLAGS.max_lr, step_size=2740, mode='exp_range', gamma=0.99994)
    matcher = TextMatcher(FLAGS.model_name, FLAGS.vocab_file, FLAGS.do_lower_case, FLAGS.max_seq_len)
    
    model_builder = ModelBuilder(
        model_name = FLAGS.model_name,
        max_len = FLAGS.max_seq_len,
        input_dim = FLAGS.input_dim,
        max_features = FLAGS.max_features,
        units = FLAGS.units,
        num_filter = FLAGS.num_filter
        )
    
    if FLAGS.is_fixed_emb:
        model_builder.set_embedding_matrix(emb_matrix)

    model = model_builder.build_model()
    
    print(model.summary())

    if FLAGS.do_train:
        train_example = matcher.get_train_examples(FLAGS.data_dir)
        matcher.do_train(model, FLAGS.output_dir, train_example, FLAGS.epochs, FLAGS.batch_size, callback=[clr, ])

    if FLAGS.do_eval:
        dev_example = matcher.get_dev_examples(FLAGS.data_dir)
        matcher.do_eval(model, FLAGS.output_dir, dev_example, FLAGS.batch_size)

    if FLAGS.do_predict:
        test_example = matcher.get_test_examples(FLAGS.data_dir)
        matcher.do_predict(model, FLAGS.output_dir, test_example, FLAGS.batch_size)

if __name__ == '__main__':
    tf.app.run()
