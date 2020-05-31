# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import tensorflow as tf
import numpy as np
from sklearn import metrics
from keras.preprocessing.sequence import pad_sequences

import utils as utils
import tokenization

tf.logging.set_verbosity(tf.logging.INFO)

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single 
                    sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                    Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                    specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class TextMatcher(object):
    """Processor for data set """
    def __init__(self, model_name, vocab_file, do_lower_case, max_seq_len):
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.CharTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)
    
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, data_path, set_type):
        lines = self._read_tsv(data_path)
        fea_a, fea_b, y = [], [], []
        for (i, line) in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            if 2 > len(line):
                continue
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            if "test" == set_type:
                label = "0"
            else:
                if 3 > len(line):
                    continue
                label = int(line[2])
            
            id_text_a = self.tokenizer.convert_tokens_to_ids(text_a)
            id_text_b = self.tokenizer.convert_tokens_to_ids(text_b)
            fea_a.append(id_text_a)
            fea_b.append(id_text_b)
            y.append(label)
            
            if i < 5:
                tf.logging.info("*** Example in [%s] ***" % (data_path))
                tf.logging.info("guid: %s" % (guid))
                tf.logging.info("tokens_a: %s" % " ".join([tokenization.printable_text(x) for x in text_a]))
                tf.logging.info("input_ids_a: %s" % " ".join([str(x) for x in id_text_a]))
                tf.logging.info("tokens_b: %s" % " ".join([tokenization.printable_text(x) for x in text_b]))
                tf.logging.info("input_ids_b: %s" % " ".join([str(x) for x in id_text_b]))
                tf.logging.info("label: %s" % (label))

        fea_a = pad_sequences(fea_a, maxlen=self.max_seq_len)
        fea_b = pad_sequences(fea_b, maxlen=self.max_seq_len)
        y = np.array(y)
        return [[fea_a, fea_b], y]

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.txt"), "train")
    
    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.txt"), "dev")
    
    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.txt"), "test")
    
    def get_labels(self, data_dir):
        all_label = []
        for line in open(os.path.join(data_dir, "label.txt")):
            all_label.append(tokenization.convert_to_unicode(line.strip()))
        return all_label
    
    def do_train(self, model, save_dir, example_train, 
            epochs, batch_size, callback = None):
        
        if 2 > len(example_train):
            raise ValueError("illegal train example format")
        best_loss = 1000
        data_train = example_train[0]
        label_train = example_train[1]
        for epoch in range(epochs):
            classifer = model.fit(data_train, label_train,
                    batch_size = batch_size,
                    epochs = 1,
                    callbacks = callback)
            
            score_train = model.predict(data_train,
                batch_size = batch_size,
                verbose=0)
            
            loss = classifer.history['loss'][0]
            acc = classifer.history['acc'][0]
            if loss < best_loss:
                best_loss = loss
                save_file = os.path.join(save_dir, self.model_name + "_best_model.h5")
                model.save_weights(save_file)
            
            auc = metrics.roc_auc_score(label_train, score_train)
            thresholds = []
            for thresh in np.arange(0.4, 0.801, 0.01):
                thresh = np.round(thresh, 2)
                res = metrics.f1_score(label_train, (score_train > thresh).astype(int))
                thresholds.append([thresh, res])

            thresholds.sort(key=lambda x: x[1], reverse=True)
            best_thresh = thresholds[0][0]
            f1 = metrics.f1_score(label_train, (label_train > best_thresh).astype(int))
            
            tf.logging.info("[%s] epoch:%d, loss:%.6f, acc:%.6f, f1:%.6f, auc:%.6f, best threshold:%.6f" 
                    % (str(utils.get_timestamp()), epoch, loss, acc, f1, auc, best_thresh))

    def do_eval(self, model, save_dir, example_dev, batch_size):
        
        if 2 > len(example_dev):
            raise ValueError("illegal dev example format")
        data_dev = example_dev[0]
        label_dev = example_dev[1]
        model_file = os.path.join(save_dir, self.model_name + "_best_model.h5")
        if not os.path.isfile(model_file) or not os.path.getsize(model_file):
            raise ValueError("model file [%s] not ready for valiation"  % (model_file))
        model.load_weights(model_file)
        
        loss, acc = model.evaluate(data_dev, label_dev, batch_size = batch_size, verbose = 0)
        score_dev = model.predict(data_dev, batch_size = batch_size, verbose=0)
        auc = metrics.roc_auc_score(label_dev, score_dev)
        thresholds = []
        for thresh in np.arange(0.4, 0.801, 0.01):
            thresh = np.round(thresh, 2)
            res = metrics.f1_score(label_dev, (score_dev > thresh).astype(int))
            thresholds.append([thresh, res])

        thresholds.sort(key=lambda x: x[1], reverse=True)
        best_thresh = thresholds[0][0]
        f1 = metrics.f1_score(label_dev, (label_dev > best_thresh).astype(int))
        
        tf.logging.info("[%s] validation loss:%.6f, acc:%.6f, f1:%.6f, auc:%.6f, best threshold:%.6f" 
                    % (str(utils.get_timestamp()), loss, acc, f1, auc, best_thresh))

        result = {
                "eval_loss": loss, 
                "eval_acc": acc, 
                "eval_F1": f1,
                "eval_auc": auc,
                "best_thresh": best_thresh
        }
        output_eval_file = os.path.join(save_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    def do_predict(self, model, save_dir, example_test, batch_size): 
        if 2 > len(example_test):
            raise ValueError("illegal test example format")
        data_test = example_test[0]
        model_file = os.path.join(save_dir, self.model_name + "_best_model.h5")
        if not os.path.isfile(model_file) or not os.path.getsize(model_file):
            raise ValueError("model file [%s] not ready for predict"  % (model_file))
        model.load_weights(model_file)
        
        score_test = model.predict(data_test, batch_size = batch_size, verbose=0)
        output_predict_file = os.path.join(save_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            output_line = "\n".join([str(score[0]) for score in score_test])
            writer.write(output_line)
            tf.logging.info("***** Finish do predict *****")
            output_line = "\n".join([str(score[0]) for score in score_test])
            writer.write(output_line)
