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

def create_discriminator(discrim_inputs, discrim_targets, a, n_layers=2):
        #n_layers = 2
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 32, 32, in_channels * 2] => [batch, 16, 16, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 16, 16, ndf] => [batch, 8, 8, ndf * 2]
        # layer_3: [batch, 8, 8, ndf * 2] => [batch, 4, 4, ndf * 4]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 2
                #stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                #normalized = batchnorm(convolved)
                # No BatchNorml in WGAN-GP critic
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)

        # layer_4: fully connected [batch, 4, 4, ndf * 4] => [batch, 1,1,1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            rinput = tf.reshape(rectified, [-1, 4*4*4*a.ndf])
            # there is no non-linearity
            output = discrim_fc(rinput, out_channels=1)
            #output = tf.sigmoid(convolved)
            layers.append(output)

        return tf.reshape(output,[-1])
