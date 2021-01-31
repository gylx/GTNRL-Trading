#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:59:11 2020

@author: yaoleixu
"""

# ============================================================================
# Import libraries
import numpy as np
from tensorflow.keras import activations, initializers, constraints
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
tf.random.set_seed(0)


# ============================================================================
# Special Multi-Graph Tensor Network Model
class SpecialMultiGraphTensorNetwork (tf.keras.layers.Layer):
    
    # Initialize terms
    def __init__(self, out_features, graphs_list, bias_bool=True, **kwargs):
        
        '''
        ------
        Inputs
        ------
        out_features: (int) number of feature maps
        graphs_list: (list) list of graph adjacency matrices
        bias_bool: (bool) if use bias vector or not
        
        '''

        # Save input variables
        self.out_features = out_features
        self.graphs_list = graphs_list.copy()
        self.bias_bool = bias_bool
        
        # Useful variables
        self.n_graphs = len(graphs_list)
        self.graphs_shapes = [g.shape[0] for g in graphs_list]
        
        super(SpecialMultiGraphTensorNetwork, self).__init__(**kwargs)

    # Define weights
    def build(self, input_shape):
        
        # Graph adjacency matrices: I+D^{0.5}AD^{0.5}
        for i in range(self.n_graphs):
            self.graphs_list[i] = tf.constant(tf.cast(self.graphs_list[i]+np.eye(self.graphs_shapes[i]), tf.float32))

        # Feature map kernel
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.out_features),
                                      initializer='uniform',
                                      trainable=True)
                                        
        # Bias tensor
        if self.bias_bool:
            self.bias = self.add_weight(name='bias',
                                        shape=self.graphs_shapes + [self.out_features],
                                        initializer='uniform',
                                        trainable=True)
                        
        # Be sure to call this at the end
        super(SpecialMultiGraphTensorNetwork, self).build(input_shape)

    # Forward pass
    def call(self, x):
                
        # Generate feature map
        output = tf.tensordot(x, self.kernel, axes=[[-1],[0]])

        # Graph filters
        for i in range(self.n_graphs):
            output = tf.tensordot(self.graphs_list[i], output, axes=[[-1],[i+1]])
        
        # Transpose back
        transposing_list = [i for i in range(self.n_graphs, -1, -1)] + [self.n_graphs+1]
        output = tf.transpose(output, transposing_list)
        
        # Bias tensor
        if self.bias_bool: output = output + self.bias # add bias
                    
        return output

    # Compute output shape
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_features)
        

# ============================================================================
# Tensor-Train Fully-Connected Layer from Tensorizing Neural Networks
class TensorTrainLayer (tf.keras.layers.Layer):
    
    # define initial variables needed for implementation
    def __init__(self, tt_ips, tt_ops, tt_ranks, bias_bool=True, **kwargs):

        # Tensor Train Variables.
        self.tt_ips = np.array(tt_ips)
        self.tt_ops = np.array(tt_ops)
        self.tt_ranks = np.array(tt_ranks)
        self.num_dim = np.array(tt_ips).shape[0]
        self.param_n = np.sum(self.tt_ips*self.tt_ops*self.tt_ranks[1:]*self.tt_ranks[:-1])
        self.bias_bool = bias_bool
  
        super(TensorTrainLayer, self).__init__(**kwargs)

    # define weights for each core
    def build(self, input_shape):

        # Initalize weights for the TT FCL. Note that Keras will pass the optimizer directly on these core parameters
        self.cores = []
        for d in range(self.num_dim):
            if d == 0: my_shape = (self.tt_ips[d], self.tt_ops[d], self.tt_ranks[d+1])
            elif d == self.num_dim-1: my_shape = (self.tt_ranks[d], self.tt_ips[d], self.tt_ops[d])
            else: my_shape = (self.tt_ranks[d], self.tt_ips[d], self.tt_ops[d], self.tt_ranks[d+1])
            
            self.cores.append(self.add_weight(name='tt_core_{}'.format(d),
                                              shape=my_shape,
                                              initializer='uniform',
                                              trainable=True))
        
        # Bias vector
        if self.bias_bool:
            self.bias = self.add_weight(name='bias',
                                        shape=self.tt_ops,
                                        initializer='uniform',
                                        trainable=True)
        # Be sure to call this at the end
        super(TensorTrainLayer, self).build(input_shape)

    # Implementing the layer logic
    def call(self, x, mask=None):

        w = self.cores[0]
        for d in range(1, self.num_dim):
            w = tf.tensordot(w, self.cores[d], [[-1],[0]])

        output = tf.tensordot(x, w, [[i for i in range(1, 3+1)], [i for i in range(0, 2*3, 2)]])
        
        if self.bias_bool: output = output + self.bias

        return output

    # Compute input/output shapes
    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(self.tt_ops))


# ============================================================================
# Graph Convolution Neural Network (slightly modified) from
# github.com/vermaMachineLearning/keras-deep-graph-learning
class GraphCNN(Layer):

    def __init__(self,
                 output_dim,
                 num_filters,
                 graph_conv_filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphCNN, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.num_filters = num_filters
        if num_filters != int(graph_conv_filters.get_shape().as_list()[-2]/graph_conv_filters.get_shape().as_list()[-1]):
            raise ValueError('num_filters does not match with graph_conv_filters dimensions.')
        self.graph_conv_filters = graph_conv_filters

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer.__name__ = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):

        self.input_dim = input_shape[-1]
        kernel_shape = (self.num_filters * self.input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, input):

        # # 3D tensor input: (batch_size x n_nodes x n_features)
        # output = graph_conv_op(input, self.num_filters, self.graph_conv_filters, self.kernel)
        output = tf.tensordot(self.graph_conv_filters, input, [1, 1])
        output = tf.tensordot(output, self.kernel, [2, 0])
        output = tf.transpose(output, [1, 0, 2])
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim)
        return output_shape