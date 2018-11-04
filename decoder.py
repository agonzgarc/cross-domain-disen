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

def create_generator_decoder(sR, eR, generator_outputs_channels, a, noise = True):

    # Do tiling here
    batch_size = sR.shape[0]
    latent_dim = eR.shape[-1]
    image_size = sR.shape[1]

    z = tf.reshape(eR, [batch_size, 1, 1, latent_dim])
    z = tf.tile(z, [1, image_size, image_size, 1])

    initial_input = tf.concat([sR,z],axis=3)

    # Add noise only at train time
    if a.mode == "train":
        if noise:
            inoise = tf.random_normal(initial_input.shape,mean=0.0,stddev=a.noise)
            initial_input += inoise

        layer_specs = [
                (a.ngf * 8, 0.5),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
                (a.ngf * 4, 0.5),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
                (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

    else:
        layer_specs = [
                (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
                (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
                (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

    num_encoder_layers = 5
    layers =[]

    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                # Use here combination of shared and exclusive
                input =  initial_input
            else:
                input = layers[-1]

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels, a)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        # No skip connections
        input = layers[-1]
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels, a)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

