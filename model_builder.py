#!/bin/env/python
#-*- coding: utf-8 -*-

from keras.optimizers import *
from keras.initializers import glorot_normal,orthogonal
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
import numpy as np

from keras_ordered_neurons import ONLSTM
from keras_multi_head import MultiHeadAttention

import basic_model as BM

def abs_layer(tensor):
    return Lambda(K.abs)(tensor)
    
def text_cnn(embedding_matrix = np.zeros(0), max_len = 21, input_dim = 111857, max_features = 50000, units = 150, num_filter = 128):
    ''' A Sensitivity Analysis of Convolutional Neural Networks for Sentence Classification (Text-CNN) '''
    inp1 = Input(shape=(max_len,), dtype='int32')
    inp2 = Input(shape=(max_len,), dtype='int32')
    x1 = Embedding(max_features, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(inp1)
    x1 = SpatialDropout1D(rate=0.24)(x1)

    x2 = Embedding(max_features, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(inp2)
    x2 = SpatialDropout1D(rate=0.24)(x2)

    conv1 = []
    conv2 = []
    for kernel_size in [2, 3, 4]:
        c1 = Conv1D(100, kernel_size, activation='relu')(x1)
        c1 = GlobalMaxPooling1D()(c1)
        conv1.append(c1)

        c2 = Conv1D(100, kernel_size, activation='relu')(x2)
        c2 = GlobalMaxPooling1D()(c2)
        conv2.append(c2)

    output1 = Concatenate()(conv1)
    output2 = Concatenate()(conv2)

    diff = abs_layer(Subtract()([output1, output2]))
    mul = Multiply()([output1, output2])

    outputp = Concatenate()([diff, mul])
    outputp = Dense(units)(outputp)
    outputp = ReLU()(outputp)
    finalout = Dense(1, activation='sigmoid')(outputp)
    model = Model(inputs=[inp1, inp2], outputs=finalout)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
   
def esim(embedding_matrix = np.zeros(0), max_len = 21, input_dim = 111857, max_features = 50000, units = 150, num_filter = 128):
    ''' Enhanced LSTM for Natural Language Inference (ESIM) '''
    embedding_layer = Embedding(
        max_features,
        embedding_matrix.shape[1],
        weights = [embedding_matrix],
        input_length = max_len,
        trainable = False)

    input_q1_layer = Input(shape=(max_len,), dtype='int32', name='q1')
    input_q2_layer = Input(shape=(max_len,), dtype='int32', name='q2')

    embedding_sequence_q1 = BatchNormalization(axis=2)(embedding_layer(input_q1_layer))
    embedding_sequence_q2 = BatchNormalization(axis=2)(embedding_layer(input_q2_layer))

    final_embedding_sequence_q1 = SpatialDropout1D(0.25)(embedding_sequence_q1)
    final_embedding_sequence_q2 = SpatialDropout1D(0.25)(embedding_sequence_q2)

    rnn_layer_q1 = Bidirectional(LSTM(units, return_sequences=True))(final_embedding_sequence_q1)
    rnn_layer_q2 = Bidirectional(LSTM(units, return_sequences=True))(final_embedding_sequence_q2)

    attention = Dot(axes=-1)([rnn_layer_q1, rnn_layer_q2])
    w_attn_1 = Softmax(axis=1)(attention)
    w_attn_2 = Permute((2, 1))(Softmax(axis=2)(attention))
    align_layer_1 = Dot(axes=1)([w_attn_1, rnn_layer_q1])
    align_layer_2 = Dot(axes=1)([w_attn_2, rnn_layer_q2])

    subtract_layer_1 = subtract([rnn_layer_q1, align_layer_1])
    subtract_layer_2 = subtract([rnn_layer_q2, align_layer_2])

    multiply_layer_1 = multiply([rnn_layer_q1, align_layer_1])
    multiply_layer_2 = multiply([rnn_layer_q2, align_layer_2])

    m_q1 = concatenate([rnn_layer_q1, align_layer_1, subtract_layer_1, multiply_layer_1])
    m_q2 = concatenate([rnn_layer_q2, align_layer_2, subtract_layer_2, multiply_layer_2])

    v_q1_i = Bidirectional(LSTM(units, return_sequences=True))(m_q1)
    v_q2_i = Bidirectional(LSTM(units, return_sequences=True))(m_q2)

    avgpool_q1 = GlobalAveragePooling1D()(v_q1_i)
    avgpool_q2 = GlobalAveragePooling1D()(v_q2_i)
    maxpool_q1 = GlobalMaxPooling1D()(v_q1_i)
    maxpool_q2 = GlobalMaxPooling1D()(v_q2_i)

    merged_q1 = concatenate([avgpool_q1, maxpool_q1])
    merged_q2 = concatenate([avgpool_q2, maxpool_q2])

    final_v = BatchNormalization()(concatenate([merged_q1, merged_q2]))
    output = Dense(units=256, activation='relu')(final_v)
    output = BatchNormalization()(output)
    output = Dropout(0.2)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[input_q1_layer, input_q2_layer], output=output)
    adam_optimizer = Adam(lr=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['binary_crossentropy', 'accuracy'])
    return model

def transformer(embedding_matrix = np.zeros(0), max_len = 21, input_dim = 111857, max_features = 50000, units = 150, num_filter = 128):
    inp1 = Input(shape=(max_len,), dtype='int32')
    inp2 = Input(shape=(max_len,), dtype='int32')
    
    x1 =  Embedding(input_dim=input_dim, output_dim=units*2, input_length=max_len, trainable=True)(inp1)
    x1 = SpatialDropout1D(rate=0.24)(x1)

    x2 =  Embedding(input_dim=input_dim, output_dim=units*2, input_length=max_len, trainable=True)(inp2)
    x2 = SpatialDropout1D(rate=0.24)(x2)

    x11 = MultiHeadAttention(head_num=3, name='Multi-Head1',)(x1)
    x12 = MultiHeadAttention(head_num=3, name='Multi-Head2',)(x2)

    x_first = Add()([x1,x11])
    x_second = Add()([x2,x12])

    x_first = BatchNormalization()(x_first)
    x_second = BatchNormalization()(x_second)

    x21 = Dense(units*2)(x_first)
    x22 = Dense(units * 2)(x_second)

    x_final1 = Add()([x_first,x21])
    x_final2 = Add()([x_second,x22])

    x_final1 = BatchNormalization()(x_final1)
    x_final2 = BatchNormalization()(x_final2)

    x_final1 = Flatten()(x_final1)
    x_final2 = Flatten()(x_final2)

    diff = abs_layer(Subtract()([x_final1, x_final2]))
    mul = Multiply()([x_final1, x_final2])

    outputp = Concatenate()([x_final1,x_final2,diff, mul])

    outputp = Dense(units*2)(outputp)
    outputp = BatchNormalization()(outputp)
    outputp = ReLU()(outputp)

    finalout = Dense(1, activation='sigmoid')(outputp)
    model = Model(inputs=[inp1, inp2], outputs=finalout)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def lstm(embedding_matrix = np.zeros(0), max_len = 21, input_dim = 111857, max_features = 50000, units = 150, num_filter = 128):
    inp1 = Input(shape=(max_len,), dtype='int32')
    inp2 = Input(shape=(max_len,), dtype='int32')
    x1 = Embedding(input_dim=input_dim, output_dim=units*2, input_length=max_len, trainable=True)(inp1)
    x1 = SpatialDropout1D(rate=0.24)(x1)

    x2 = Embedding(input_dim=input_dim, output_dim=units*2, input_length=max_len, trainable=True)(inp2)
    x2 = SpatialDropout1D(rate=0.24)(x2)

    x1 = Bidirectional(
        layer = LSTM(
            units,
            return_sequences = True,
            kernel_initializer = glorot_normal(seed = 1029),
            recurrent_initializer = orthogonal(gain = 1.0, seed = 1029)
        ),
        name='bidirectional_lstm1')(x1)
    x2 = Bidirectional(
        layer = LSTM(
            units,
            return_sequences = True,
            kernel_initializer = glorot_normal(seed = 1029),
            recurrent_initializer = orthogonal(gain = 1.0, seed = 1029)
        ),
        name='bidirectional_lstm2')(x2)

    x1 = Reshape((max_len, units * 2))(x1)
    x2 = Reshape((max_len, units * 2))(x2)

    x1 = Flatten()(x1)
    x2 = Flatten()(x2)

    diff = abs_layer(Subtract()([x1, x2]))
    mul = Multiply()([x1, x2])

    outputp = Concatenate()([x1, x2, diff, mul])

    outputp = Dense(int(units * 2))(outputp)
    outputp = BatchNormalization()(outputp)
    outputp = ReLU()(outputp)
    finalout = Dense(1, activation='sigmoid')(outputp)

    model = Model(inputs=[inp1, inp2], outputs=finalout)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def attention(embedding_matrix = np.zeros(0), max_len = 21, input_dim = 111857, max_features = 50000, units = 150, num_filter = 128):
    inp1 = Input(shape=(max_len,), dtype='int32')
    inp2 = Input(shape=(max_len,), dtype='int32')
    x1 = Embedding(max_features, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(inp1)
    x1 = SpatialDropout1D(rate=0.24)(x1)

    x2 = Embedding(max_features, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(inp2)
    x2 = SpatialDropout1D(rate=0.24)(x2)

    output1 = BM.Attention(max_len)(x1)
    output2 = BM.Attention(max_len)(x2)

    diff = abs_layer(Subtract()([output1, output2]))
    mul = Multiply()([output1, output2])

    outputp = Concatenate()([diff, mul])
    outputp = Dense(units, activation='relu')(outputp)
    finalout = Dense(1, activation='sigmoid')(outputp)
    model = Model(inputs=[inp1, inp2], outputs=finalout)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def capsule(embedding_matrix = np.zeros(0), max_len = 21, input_dim = 111857, max_features = 50000, units = 150, num_filter = 128):
    inp1 = Input(shape=(max_len,), dtype='int32')
    inp2 = Input(shape=(max_len,), dtype='int32')
    x1 = Embedding(input_dim=input_dim, output_dim=units*2, input_length=max_len, trainable=True)(inp1)
    x1 = SpatialDropout1D(rate=0.24)(x1)

    x2 = Embedding(input_dim=input_dim, output_dim=units*2, input_length=max_len, trainable=True)(inp2)
    x2 = SpatialDropout1D(rate=0.24)(x2)

    output1 = BM.Capsule(num_capsule=5, dim_capsule=16)(x1)
    output2 = BM.Capsule(num_capsule=5, dim_capsule=16)(x2)

    output1 = Flatten()(output1)
    output2 = Flatten()(output2)
    diff = abs_layer(Subtract()([output1, output2]))
    mul = Multiply()([output1, output2])

    outputp = Concatenate()([output1, output2, diff, mul])

    outputp = Dense(int(units/2))(outputp)
    outputp = BatchNormalization()(outputp)
    outputp = ReLU()(outputp)
    finalout = Dense(1, activation='sigmoid')(outputp)

    model = Model(inputs=[inp1, inp2], outputs=finalout)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

models = {
    "TEXT_CNN": text_cnn,
    "ESIM": esim,
    "TRANSFORMER": transformer,
    "LSTM": lstm,
    "ATT": attention,
    "CAPSULE": capsule,
}

class ModelBuilder(object):
    def __init__(
            self,
            model_name, 
            max_len, 
            input_dim,
            max_features,
            units,
            num_filter):
        self.model_name = model_name
        self.max_len = max_len
        self.input_dim = input_dim
        self.max_features = max_features
        self.units = units
        self.num_filter = num_filter
        self.embedding_matrix = np.zeros(0)

    def set_embedding_matrix(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix

    def build_model(self):
        return models[self.model_name](
                embedding_matrix = self.embedding_matrix,
                max_len = self.max_len,
                input_dim = self.input_dim,
                max_features = self.max_features,
                units = self.units,
                num_filter = self.num_filter)

if __name__ == '__main__':
    ''' test class ModelBuilder'''
    model_builder = ModelBuilder("TEXT_CNN", None, 21, 111857, 50000, 150, 128)
    model_builder.build_model()
