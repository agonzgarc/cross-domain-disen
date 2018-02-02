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

def create_generator_encoder(generator_inputs, a):
    layers = []

    # encoder_1: [batch, 32, 32, in_channels] => [batch, 16, 16, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf, a)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 16, 16, ngf] => [batch, 8, 8, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 8, 8, ngf * 2] => [batch, 4, 4, ngf * 4]
        a.ngf * 4, # encoder_4: [batch, 4, 4, ngf * 4] => [batch, 2, 2, ngf * 4]
        a.ngf * 4, # encoder_5: [batch, 2, 2, ngf * 4] => [batch, 1, 1, ngf * 4]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, a)
            output = batchnorm(convolved)
            layers.append(output)

    num_channels = int(output.shape[3])
    # Hard-code it for now, fix it later
    #num_channels = 512
    sR = output[:,:,:,:int(num_channels/2)]
    eR = output[:,:,:,int(num_channels/2):]

    return sR,eR,layers

