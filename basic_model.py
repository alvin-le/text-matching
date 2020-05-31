#!/bin/env/python
#-*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import glorot_normal, orthogonal

from keras import backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import *

class Attention(Layer):
    def __init__(self,
                 step_dim,
                 W_regularizer = None,
                 b_regularizer = None,
                 W_constraint = None,
                 b_constraint = None,
                 bias=True,
                 **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            '{}_W'.format(self.name),
            shape = (input_shape[-1],),
            initializer = self.init,
            regularizer = self.W_regularizer,
            constraint = self.W_constraint)
        
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(
                '{}_b'.format(self.name),
                shape = (input_shape[1],),
                initializer = 'zero',
                regularizer = self.b_regularizer,
                constraint = self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

class CrossAttention(Layer):
    def __init__(
            self,
            step_dim,
            Wc_regularizer=None,
            W1_regularizer=None,
            W2_regularizer=None,
            Wc_constraint=None,
            W1_constraint=None,
            W2_constraint=None,
            bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.Wc_regularizer = regularizers.get(Wc_regularizer)
        self.W1_regularizer = regularizers.get(W1_regularizer)
        self.W2_regularizer = regularizers.get(W2_regularizer)

        self.Wc_constraint = constraints.get(Wc_constraint)
        self.W1_constraint = constraints.get(W1_constraint)
        self.W2_constraint = constraints.get(W2_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Cross_Attention, self).__init__(**kwargs)

    def build(self, input_shape1):
        input_shape1 = input_shape1[0]
        assert len(input_shape1) == 3

        self.Wc = self.add_weight(
            '{}_Wc'.format(self.name),
            shape = (input_shape1[-1], input_shape1[-1]),
            initializer = self.init,
            regularizer = self.Wc_regularizer,
            constraint = self.Wc_constraint)

        self.W1 = self.add_weight(
            '{}_W1'.format(self.name),
            shape = (input_shape1[-1],),
            initializer = self.init,
            regularizer = self.W1_regularizer,
            constraint = self.W1_constraint)

        self.W2 = self.add_weight(
            '{}_W2'.format(self.name),
            shape = (input_shape1[-1],),
            initializer = self.init,
            regularizer = self.W2_regularizer,
            constraint = self.W2_constraint)

        self.features_dim = input_shape1[-1]
        self.batch = input_shape1[0]
        self.step_dim = input_shape1[1]
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, input, mask=None):
        self.batch = -1
        x1 = K.reshape(input[0],[self.batch,self.step_dim,self.features_dim])
        x2 = K.reshape(input[1],[self.batch,self.step_dim,self.features_dim])
        features_dim = self.features_dim

        #(batch*max_len*2h)*(2h*2h)  = batch*max_len*2h
        x1_dot_w = K.dot(K.reshape(x1, (-1,self.step_dim, features_dim)),self.Wc)
        #(batch*max_len*2h)*(batch*2h*max_len) = batch*max_len*max_len
        x1_dot_w_dot_x2 = K.batch_dot(x1_dot_w,K.reshape(x2,(-1,features_dim,self.step_dim)),axes=[2,1])
        # (batch*max_len*2h) * (2h*1) = batch*max_len*1
        x1_dot_w1 = K.dot(x1,K.reshape(self.W1,(features_dim,1)))
        # batch*max_len*max_len
        x1_dot_w1 = K.repeat_elements(x1_dot_w1,self.step_dim,2)
        # (batch*max_len*2h) * (2h*1) = batch*max_len*1
        x2_dot_w2 = K.dot(x2,K.reshape(self.W2,(features_dim,1)))
        # 1*maxlen
        x2_dot_w2 = K.reshape(x2_dot_w2, (-1,1,self.step_dim))
        # max_len*max_len
        x2_dot_w2 = K.repeat_elements(x2_dot_w2,self.step_dim,1)
        # max_len * max_len
        A = Add()([x1_dot_w_dot_x2,x1_dot_w1])
        A = Add()([A,x2_dot_w2])

        # get attention matrix A
        # first raw represent attention score of the first element respect to all element in x2
        x1_to_x2all = K.softmax(A,axis=1) # softmax by raw
        x2_to_x1all = K.softmax(A,axis=2) # softmax by column

        # attention to x1 (max_len * max_len) * (max_len * 2h)
        f_x1 = K.batch_dot(x1_to_x2all,x2,axes=[2,1])
        f_x1 = K.reshape(f_x1,(-1,self.step_dim,self.features_dim))
        # attention to x2
        f_x2 = K.batch_dot(x2_to_x1all,x1,axes=[1,1])
        f_x2 = K.reshape(f_x2, (-1,self.step_dim, self.features_dim))
        return [f_x1,f_x2]

    def compute_output_shape(self, input_shape1):
        return [input_shape1[0],input_shape1[1]]

class SelfAttention(Layer):
    def __init__(
        self,
        step_dim,
        Wc_regularizer = None,
        W1_regularizer = None,
        W2_regularizer = None,
        Wc_constraint = None, 
        W1_constraint = None,
        W2_constraint = None,
        bias=True,
        **kwargs):
        
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.Wc_regularizer = regularizers.get(Wc_regularizer)
        self.W1_regularizer = regularizers.get(W1_regularizer)

        self.Wc_constraint = constraints.get(Wc_constraint)
        self.W1_constraint = constraints.get(W1_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape1):
        assert len(input_shape1) == 3

        self.Wc = self.add_weight(
            '{}_Wsc'.format(self.name),
            shape = (input_shape1[-1],input_shape1[-1]),
            initializer = self.init,
            regularizer = self.Wc_regularizer,
            constraint = self.Wc_constraint)

        self.W1 = self.add_weight(
            '{}_Ws1'.format(self.name),
            shape = (input_shape1[-1],),
            initializer = self.init,
            regularizer = self.W1_regularizer,
            constraint = self.W1_constraint)
            
        self.features_dim = input_shape1[-1]
        self.batch = input_shape1[0]
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, input,mask=None):
        self.batch = -1
        x1 = K.reshape(input,[self.batch,self.step_dim,self.features_dim])

        features_dim = self.features_dim

        #(batch*max_len*2h)*(2h*2h)  = batch*max_len*2h
        x1_dot_w = K.dot(K.reshape(x1, (-1,self.step_dim,features_dim)),self.Wc)

        #(batch*max_len*2h)*(batch*2h*max_len) = batch*max_len*max_len
        x1_dot_w_dot_x2 = K.batch_dot(x1_dot_w,K.reshape(x1,(-1,features_dim,self.step_dim)),axes=[2,1])

        # (batch*max_len*2h) * (2h*1) = batch*max_len*1
        x1_dot_w1 = K.dot(K.reshape(x1,(-1,self.step_dim,self.features_dim)),K.reshape(self.W1,(features_dim,-1)))
        # batch*max_len*max_len
        x1_dot_w1 = K.repeat_elements(x1_dot_w1,self.step_dim,2)

        # batch * max_len * max_len
        A = Add()([x1_dot_w_dot_x2,x1_dot_w1])

        # batch * maxlen * maxlen
        x1_to_x2all = K.softmax(A,axis=1)

        f_x1 = K.batch_dot(x1_to_x2all,K.reshape(x1,(-1,self.step_dim,self.features_dim)),axes=[2,1])
        f_x1 = K.reshape(f_x1,(-1,self.step_dim,self.features_dim))

        return f_x1

    def compute_output_shape(self, input_shape1):
        return input_shape1

class ConvInteration(Layer):
    def __init__(
        self, 
        k,
        pi = 1,
        Wu_regularizer = None,
        P_regularizer = None,
        Q_regularizer = None,
        B_regularizer = None,
        Wu_constraint=None,
        P_constraint=None,
        Q_constraint=None,
        B_constraint=None,
        bias=True, 
        **kwargs):

        self.init = initializers.get('glorot_uniform')

        self.Wu_regularizer = regularizers.get(Wu_regularizer)
        self.P_regularizer = regularizers.get(P_regularizer)
        self.Q_regularizer = regularizers.get(Q_regularizer)
        self.B_regularizer = regularizers.get(B_regularizer)

        self.Wu_constraint = constraints.get(Wu_constraint)
        self.P_constraint = constraints.get(P_constraint)
        self.Q_constraint = constraints.get(Q_constraint)
        self.B_constraint = constraints.get(B_constraint)

        self.k = k
        self.pi = pi
        super(ConvInteration, self).__init__(**kwargs)

    def build(self, input_shape1):
        assert len(input_shape1) == 3
        self.step_dim = input_shape1[1]

        self.Wu = self.add_weight(
            '{}_Wu'.format(self.name),
            shape = (input_shape1[-1], input_shape1[-1]),
            initializer = self.init,
            regularizer = self.Wu_regularizer,
            constraint = self.Wu_constraint)

        self.P = self.add_weight(
            '{}_P'.format(self.name),
            shape = (int(input_shape1[-1]/self.k), input_shape1[-1]),
            initializer = self.init,
            regularizer = self.P_regularizer,
            constraint = self.P_constraint)

        self.Q = self.add_weight(
            '{}_Q'.format(self.name),
            shape = (input_shape1[-1], self.pi*input_shape1[-1]),
            initializer = self.init,
            regularizer = self.Q_regularizer,
            constraint = self.Q_constraint)

        self.B = self.add_weight(
            '{}_B'.format(self.name),
            shape = (int(input_shape1[-1]/self.k), self.pi*input_shape1[-1]),
            initializer = self.init,
            regularizer = self.B_regularizer,
            constraint = self.B_constraint)

        self.features_dim = input_shape1[-1]
        self.batch = -1
        self.built = True

    def compute_output_shape(self, input_shape1):
        return (input_shape1[0],input_shape1[-1] ,self.pi*input_shape1[-1])

    def call(self, input,mask=None):

        x1 = K.reshape(input,[self.batch,self.step_dim,self.features_dim])

        # batch * step* d
        Ux = K.relu(K.reshape(K.dot(x1,self.Wu),[self.batch,self.step_dim,self.features_dim]))
        # k_max_pooling batch*d*k
        z = KMaxPooling(self.k)(Ux)
        z = K.reshape(z,[self.batch,self.k,self.features_dim])

        output = []

        # d * d/k
        self.P = K.reshape(self.P,[self.features_dim,-1])
        self.Q = K.reshape(self.Q, [self.features_dim, -1])
        for i in range(self.k):
            temp = z[:,i,:]
            temp = K.reshape(temp,[self.batch,self.features_dim])
            zi = tf.matrix_diag(temp)

            # batch*d*d * (d * d/k) = batch * d * d/k
            zi_mul_p = K.dot(zi,self.P)
            # batch * d/k * d
            zi_mul_p = K.reshape(zi_mul_p,[self.batch,int(self.features_dim/self.k),self.features_dim])

            # (batch * d/k * d) * (d * pid) = batch * d/k * pid
            p_mul_zi_mul_q = K.dot(zi_mul_p,self.Q)
            out = K.relu(K.bias_add(p_mul_zi_mul_q,self.B))
            output.append(out)

        w_concat = K.concatenate(output,axis=1)
        return w_concat

class BatchDot(Layer):
    def __init__(self, axes, **kwargs):
        self.axes = axes
        super(BatchDot, self).__init__(**kwargs)

    def build(self, input_shape1):
        assert len(input_shape1[0]) == 3

    def compute_output_shape(self, input_shape):
        a1 = self.axes[0]
        a2 = self.axes[1]
        input_shape1 = input_shape[0]
        input_shape2 = input_shape[1]
        return (input_shape1[0], input_shape1[a2],input_shape2[a1])

    def call(self, input1):
        x1 = input1[0]
        x2 = input1[1]
        batch_dot = K.batch_dot(x1, x2, axes=self.axes)
        return batch_dot

class Capsule(Layer):
    ''' A Capsule Implement with Pure Keras '''
    def __init__(self, num_capsule, dim_capsule, routings=4, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.init = initializers.get('glorot_uniform')
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(
                name = 'capsule_kernel',
                shape = (1, int(input_dim_capsule), int(self.num_capsule * self.dim_capsule)),
                initializer = 'glorot_uniform',
                trainable = True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(
                name = 'capsule_kernel',
                shape = (input_num_capsule, input_dim_capsule, self.num_capsule * self.dim_capsule),
                initializer = 'glorot_uniform',
                trainable = True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            o = K.batch_dot(c, u_hat_vecs, [2, 2])
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                b = K.batch_dot(o, u_hat_vecs, [2, 3])
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

class Reverse(Layer):
    ''' K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).TensorFlow backend. '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

    def call(self, inputs):
        reverse = K.reverse(inputs,axes=1)
        # return flattened output
        return reverse
        
class KMaxPooling(Layer):
    ''' K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).TensorFlow backend. '''
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2] ,self.k)

    def call(self, inputs):
        # extract top_k, returns two tensors [values, indices]
        inputs = tf.transpose(inputs,[0,2,1])
        top_k = tf.nn.top_k(inputs, k=self.k, sorted=False, name=None)[0]
        # return flattened output
        return top_k

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ''' define our own softmax function instead of K.softmax '''
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

class Attention_multihead(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention_multihead, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        self.WQ = self.add_weight(name='WQ',
                                  shape=(int(input_shape[0][-1]), self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(int(input_shape[1][-1]), self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(int(input_shape[2][-1]), self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention_multihead, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # no mask if only Q_seq/K_seq/V_seq
        # do mask if Q_seq/K_seq/V_seq/Q_len/V_len
        if 3 == len(x):
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif 5 == len(x):
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # linear tranformation on Q/K/V
        
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, Q_seq.shape[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K_seq.shape[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, V_seq.shape[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))  # A (?, 70, 70, 2)
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, O_seq.shape[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
