from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

from ops import *

@tf.RegisterGradient("ReverseGrad")
def _reverse_grad(unused_op, grad):
    return -1.0*grad

def create_generator_decoder_exclusive(eR, layers, generator_outputs_channels, a):

    if a.red_images:
        layer_specs = [
            (a.ngf * 4, 0.5),   # decoder_5: [batch, 1, 1, ngf * 4] => [batch, 2, 2, ngf * 4 * 2]
            (a.ngf * 4, 0.5),   # decoder_7: [batch, 2, 2, ngf * 4 * 2] => [batch, 4, 4, ngf * 4 * 2]
            # We could remove the dropout from the following one, maybe
            (a.ngf * 3, 0.5),   # decoder_6: [batch, 4, 4, ngf * 4 * 2] => [batch, 8, 8, ngf * 2 * 2]
            (a.ngf, 0.0),   # decoder_5: [batch, 8, 8, ngf * 2 * 2] => [batch, 16, 16, ngf  * 2]
        ]

        # Hard-coded, find better way to pass this value
        num_encoder_layers = 5
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer

                    # Use here combination of shared and exclusive
                    #input = eR
                    g = tf.get_default_graph()
                    with g.gradient_override_map({"Identity": "ReverseGrad"}):
                        rectified = tf.identity(eR)
                else:
                    #input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                    input = layers[-1]
                    rectified = tf.nn.relu(input)

                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = gen_deconv_upsample(rectified, out_channels, a)
                output = batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 16, 16, ngf] => [batch, 32, 32, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            # This is for the skip connection
            input = layers[-1]
            #input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = gen_deconv_upsample(rectified, generator_outputs_channels, a)
            output = tf.tanh(output)
            layers.append(output)


    else:
        layer_specs = [
            (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            #(a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]


        # Hard-coded, find better way to pass this value
        num_encoder_layers = 8
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer

                    # Use here combination of shared and exclusive
                    #input = eR
                    g = tf.get_default_graph()
                    with g.gradient_override_map({"Identity": "ReverseGrad"}):
                        input = tf.identity(eR)
                else:
                    #input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                    input = layers[-1]

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = gen_deconv(rectified, out_channels, a)
                output = batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 16, 16, ngf] => [batch, 32, 32, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            # This is for the skip connection
            input = layers[-1]
            #input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = gen_deconv(rectified, generator_outputs_channels, a)
            output = tf.tanh(output)
            layers.append(output)

    return layers[-1]

