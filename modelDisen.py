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
from generatorDecoderMNIST import *
from generatorEncoderMNIST import *
#from generatorNoSkip import *
#from discriminator import *
from discriminatorWGANGP import *

import pdb

EPS = 1e-12
CROP_SIZE = 32
LOSS = 'wgan-gp'
LAMBDA = 10
OUTPUT_DIM = 3072 # 32X32X3

Model = collections.namedtuple("Model", "outputsX2Y, outputsY2X,\
                               outputsX2Yp, outputsY2Xp,\
                               auto_outputX, auto_outputY\
                               predict_realX2Y, predict_realY2X,\
                               predict_fakeX2Y, predict_fakeY2X,\
                               im_swapped_Y,sel_auto_Y\
                               im_swapped_X,sel_auto_X\
                               discrimX2Y_loss, discrimY2X_loss,\
                               discrimX2Y_grads_and_vars, discrimY2X_grads_and_vars,\
                               genX2Y_loss_GAN, genY2X_loss_GAN,\
                               genX2Y_loss_L1, genY2X_loss_L1,\
                               genX2Y_grads_and_vars, genY2X_grads_and_vars,\
                               autoencoderX_loss, autoencoderY_loss,\
                               autoencoderX_grads_and_vars, autoencoderY_grads_and_vars,\
                               feat_recon_loss, feat_recon_grads_and_vars,\
                               train, train_disc")

def swapBackground(sR, eR, auto_output, which_direction, a):
        bkg_ims_idx = tf.random_uniform([a.batch_size],minval=0,maxval=a.batch_size,dtype=tf.int32)
        swapScoreBKG = 0
        sR_Swap = [] #tf.zeros_like(sR_Y2X)
        eR_Swap = [] #tf.zeros_like(eR_Y2X)
        sel_auto = [] #tf.zeros_like(auto_outputY)
        #im_swapped = []

        for i in range(0,a.batch_size):
            s_curr = tf.reshape(sR[i,:],[sR.shape[1],sR.shape[2],sR.shape[3]])

            #print('I:'+str(i)+' paired with:'+str(bkg_ims_idx[i]))
            # Image to swap cannot be current image
            with tf.Session() as sess:
                while bkg_ims_idx[i].eval() == i:
                    # Re-do whole batch, it doesn't matter if previous are different
                    bkg_ims_idx = tf.random_uniform([a.batch_size],minval=0,maxval=a.batch_size,dtype=tf.int32)

            ex_rnd = tf.reshape(eR[bkg_ims_idx[i],:],[eR.shape[1],eR.shape[2],eR.shape[3]])

            sR_Swap.append(s_curr)
            eR_Swap.append(ex_rnd)

            sel_auto.append(auto_output[bkg_ims_idx[i],:])

        #pdb.set_trace()
        with tf.variable_scope("generator" + which_direction + "_decoder", reuse=True):
                    out_channels = int(auto_output.get_shape()[-1])
                    im_swapped = create_generator_decoder(tf.stack(sR_Swap),
                                                          tf.stack(eR_Swap), [], out_channels, a)

        swapScoreBKG = tf.reduce_mean(tf.abs(auto_output[:,:4,:4,:] - im_swapped[:,:4,:4,:]))


        return swapScoreBKG, im_swapped, tf.stack(sel_auto)


def create_model(inputsX, inputsY, a):

    # Target for inputsX is inputsY and vice versa
    targetsX = inputsY
    targetsY = inputsX

    with tf.variable_scope("generatorX2Y_encoder"):
        #outputsX2Y = create_generator(inputsX, out_channels, a)
        sR_X2Y, eR_X2Y, layers_X2Y = create_generator_encoder(inputsX, a)

    with tf.variable_scope("generatorY2X_encoder"):
        sR_Y2X, eR_Y2X, layers_Y2X = create_generator_encoder(inputsY, a)

    #tf.summary.histogram("exclusiveX2Y", eR_X2Y)
    mean_X2Y, var_X2Y = tf.nn.moments(eR_X2Y, axes=[0,1,2])
    mean_Y2X, var_Y2X = tf.nn.moments(eR_Y2X, axes=[0,1,2])

    # One copy of the decoder for the noise input, another for the correct
    # input for the autoencoder
    with tf.name_scope("generatorX2Y_decoder_noise"):
        with tf.variable_scope("generatorX2Y_decoder"):
            out_channels = int(targetsX.get_shape()[-1])

            noise_X2Y = tf.random_normal(eR_Y2X.shape, mean=mean_Y2X,
                                        stddev=tf.sqrt(var_Y2X))

            outputsX2Y = create_generator_decoder(sR_X2Y, noise_X2Y, layers_X2Y, out_channels, a)
            #outputsX2Y = create_generator_decoder(sR_X2Y, eR_X2Y, layers_X2Y, out_channels, a)

        with tf.variable_scope("generatorX2Y_decoder", reuse=True):
            noise_X2Yp = tf.random_normal(eR_Y2X.shape, mean=mean_Y2X,
                                                stddev=tf.sqrt(var_Y2X))
            outputsX2Yp = create_generator_decoder(sR_X2Y, noise_X2Yp, layers_X2Y, out_channels, a)

    with tf.name_scope("generatorY2X_decoder_noise"):
        with tf.variable_scope("generatorY2X_decoder"):
            out_channels = int(targetsY.get_shape()[-1])

            noise_Y2X = tf.random_normal(eR_X2Y.shape, mean=mean_X2Y,
                                        stddev=tf.sqrt(var_X2Y))

            #outputsY2X = create_generator_decoder(sR_Y2X, eR_Y2X, layers_Y2X, out_channels, a)
            outputsY2X = create_generator_decoder(sR_Y2X, noise_Y2X, layers_Y2X, out_channels, a)

        with tf.variable_scope("generatorY2X_decoder",reuse=True):
            noise_Y2Xp = tf.random_normal(eR_X2Y.shape, mean=mean_X2Y,
                                        stddev=tf.sqrt(var_X2Y))
            outputsY2Xp = create_generator_decoder(sR_Y2X, noise_Y2Xp, layers_Y2X, out_channels, a)

    with tf.name_scope("autoencoderX"):
        # Use here decoder Y2X but with input from X2Y encoder
        with tf.variable_scope("generatorY2X_decoder", reuse=True):
            out_channels = int(inputsX.get_shape()[-1])
            auto_outputX = create_generator_decoder(sR_X2Y, eR_X2Y, layers_X2Y, out_channels, a)

    with tf.name_scope("autoencoderY"):
        # Use here decoder Y2X but with input from X2Y encoder
        with tf.variable_scope("generatorX2Y_decoder", reuse=True):
            out_channels = int(inputsY.get_shape()[-1])
            auto_outputY = create_generator_decoder(sR_Y2X, eR_Y2X, layers_Y2X, out_channels, a)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables

    # We will now have 2 different discriminators, one per direction, and two
    # copies of each for real/fake pairs

    with tf.name_scope("real_discriminatorX2Y"):
        with tf.variable_scope("discriminatorX2Y"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_realX2Y = create_discriminator(inputsX, targetsX, a)

    with tf.name_scope("real_discriminatorY2X"):
        with tf.variable_scope("discriminatorY2X"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_realY2X = create_discriminator(inputsY, targetsY, a)

    with tf.name_scope("fake_discriminatorX2Y"):
        with tf.variable_scope("discriminatorX2Y", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fakeX2Y = create_discriminator(inputsX, outputsX2Y, a)

    with tf.name_scope("fake_discriminatorY2X"):
        with tf.variable_scope("discriminatorY2X", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fakeY2X = create_discriminator(inputsY, outputsY2X, a)



    with tf.name_scope("image_swapper_Y"):
        _, im_swapped_Y,sel_auto_Y = swapBackground(sR_Y2X, eR_Y2X,
                                                  auto_outputY,'X2Y', a)
    with tf.name_scope("image_swapper_X"):
        _, im_swapped_X,sel_auto_X = swapBackground(sR_X2Y, eR_X2Y,
                                                  auto_outputX,'Y2X', a)

    ######### LOSSES

    if LOSS == 'wgan-gp':

        with tf.name_scope("generatorX2Y_loss"):
            genX2Y_loss_GAN = -tf.reduce_mean(predict_fakeX2Y)
            genX2Y_loss_L1 = tf.reduce_mean(tf.abs(targetsX - outputsX2Y))
            # Same parameter for loss weighting for now
            genX2Y_loss = genX2Y_loss_GAN * a.gan_weight + genX2Y_loss_L1 * a.l1_weight

        with tf.name_scope("discriminatorX2Y_loss"):
            discrimX2Y_loss = tf.reduce_mean(predict_fakeX2Y) - tf.reduce_mean(predict_realX2Y)
            alpha = tf.random_uniform(shape=[a.batch_size,1], minval=0., maxval=1.)
            differences = tf.reshape(outputsX2Y,[-1,OUTPUT_DIM])-tf.reshape(targetsX,[-1,OUTPUT_DIM])
            #pdb.set_trace()
            interpolates = tf.reshape(targetsX, [-1,OUTPUT_DIM]) + (alpha*differences)
            with tf.variable_scope("discriminatorX2Y", reuse=True):
                gradients = tf.gradients(create_discriminator(inputsX,tf.reshape(interpolates,[-1,32,32,3]),a),
                             [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                           reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)

            tf.summary.histogram("X2Y/fake_score", predict_fakeX2Y)
            tf.summary.histogram("X2Y/real_score", predict_realX2Y)
            tf.summary.histogram("X2Y/disc_loss", discrimX2Y_loss )
            tf.summary.histogram("X2Y/gradient_penaltyy", gradient_penalty)
            discrimX2Y_loss += LAMBDA*gradient_penalty

        with tf.name_scope("generatorY2X_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            genY2X_loss_GAN = -tf.reduce_mean(predict_fakeY2X)
            genY2X_loss_L1 = tf.reduce_mean(tf.abs(targetsY - outputsY2X))
            # Same parameter for loss weighting for now
            genY2X_loss = genY2X_loss_GAN * a.gan_weight + genX2Y_loss_L1 * a.l1_weight

        with tf.name_scope("discriminatorY2X_loss"):
            discrimY2X_loss = tf.reduce_mean(predict_fakeY2X) - tf.reduce_mean(predict_realY2X)
            alpha = tf.random_uniform(shape=[a.batch_size,1], minval=0., maxval=1.)
            differences = tf.reshape(outputsY2X,[-1,OUTPUT_DIM])-tf.reshape(targetsY,[-1,OUTPUT_DIM])
            interpolates = tf.reshape(targetsY,[-1,OUTPUT_DIM]) + (alpha*differences)
            with tf.variable_scope("discriminatorY2X", reuse=True):
                gradients = tf.gradients(create_discriminator(inputsY,tf.reshape(interpolates,[-1,32,32,3]),a),
                             [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                           reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            discrimY2X_loss += LAMBDA*gradient_penalty


    else:


        with tf.name_scope("discriminatorX2Y_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrimX2Y_loss = tf.reduce_mean(-(tf.log(predict_realX2Y + EPS) +
                                               tf.log(1 - predict_fakeX2Y + EPS)))

        with tf.name_scope("discriminatorY2X_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrimY2X_loss = tf.reduce_mean(-(tf.log(predict_realY2X + EPS) +
                                               tf.log(1 - predict_fakeY2X + EPS)))
        with tf.name_scope("generatorX2Y_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            genX2Y_loss_GAN = tf.reduce_mean(-tf.log(predict_fakeX2Y + EPS))
            genX2Y_loss_L1 = tf.reduce_mean(tf.abs(targetsX - outputsX2Y))
            # Same parameter for loss weighting for now
            genX2Y_loss = genX2Y_loss_GAN * a.gan_weight + genX2Y_loss_L1 * a.l1_weight

        with tf.name_scope("generatorY2X_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            genY2X_loss_GAN = tf.reduce_mean(-tf.log(predict_fakeY2X + EPS))
            genY2X_loss_L1 = tf.reduce_mean(tf.abs(targetsY - outputsY2X))
            # Same parameter for loss weighting for now
            genY2X_loss = genY2X_loss_GAN * a.gan_weight  + genY2X_loss_L1 * a.l1_weight

    with tf.name_scope("autoencoderX_loss"):
        autoencoderX_loss = a.l1_weight*tf.reduce_mean(tf.abs(auto_outputX-inputsX))

    with tf.name_scope("autoencoderY_loss"):
        autoencoderY_loss = a.l1_weight*tf.reduce_mean(tf.abs(auto_outputY-inputsY))


    with tf.name_scope("feat_recon_loss"):
        feat_recon_loss = a.l1_weight*tf.reduce_mean(tf.abs(sR_X2Y-sR_Y2X))

    ######### OPTIMIZERS


    with tf.name_scope("discriminatorX2Y_train"):
        discrimX2Y_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminatorX2Y")]
        discrimX2Y_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrimX2Y_grads_and_vars = discrimX2Y_optim.compute_gradients(discrimX2Y_loss, var_list=discrimX2Y_tvars)
        discrimX2Y_train = discrimX2Y_optim.apply_gradients(discrimX2Y_grads_and_vars)

    with tf.name_scope("discriminatorY2X_train"):
        discrimY2X_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminatorY2X")]
        discrimY2X_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrimY2X_grads_and_vars = discrimY2X_optim.compute_gradients(discrimY2X_loss, var_list=discrimY2X_tvars)
        discrimY2X_train = discrimY2X_optim.apply_gradients(discrimY2X_grads_and_vars)

    with tf.name_scope("generatorX2Y_train"):
        with tf.control_dependencies([discrimX2Y_train]):
            genX2Y_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generatorX2Y")]
            genX2Y_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            genX2Y_grads_and_vars = genX2Y_optim.compute_gradients(genX2Y_loss, var_list=genX2Y_tvars)
            genX2Y_train = genX2Y_optim.apply_gradients(genX2Y_grads_and_vars)

    with tf.name_scope("generatorY2X_train"):
        with tf.control_dependencies([discrimY2X_train]):
            genY2X_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generatorY2X")]
            genY2X_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            genY2X_grads_and_vars = genY2X_optim.compute_gradients(genY2X_loss, var_list=genY2X_tvars)
            genY2X_train = genY2X_optim.apply_gradients(genY2X_grads_and_vars)

    with tf.name_scope("autoencoderX_train"):
        autoencoderX_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorX2Y_encoder") or
                              var.name.startswith("generatorY2X_decoder")]
        autoencoderX_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        autoencoderX_grads_and_vars = autoencoderX_optim.compute_gradients(autoencoderX_loss, var_list=autoencoderX_tvars)
        autoencoderX_train = autoencoderX_optim.apply_gradients(autoencoderX_grads_and_vars)

    with tf.name_scope("autoencoderY_train"):
        autoencoderY_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorY2X_encoder") or
                              var.name.startswith("generatorX2Y_decoder")]
        autoencoderY_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        autoencoderY_grads_and_vars = autoencoderY_optim.compute_gradients(autoencoderY_loss, var_list=autoencoderY_tvars)
        autoencoderY_train = autoencoderY_optim.apply_gradients(autoencoderY_grads_and_vars)


    with tf.name_scope("feat_recon_train"):
        feat_recon_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorX2Y_encoder") or
                              var.name.startswith("generatorY2X_encoder")]
        feat_recon_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        feat_recon_grads_and_vars = feat_recon_optim.compute_gradients(feat_recon_loss, var_list=feat_recon_tvars)
        feat_recon_train = feat_recon_optim.apply_gradients(feat_recon_grads_and_vars)


    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrimX2Y_loss, discrimY2X_loss,
                               genX2Y_loss_GAN, genY2X_loss_GAN,
                               genX2Y_loss_L1, genY2X_loss_L1,
                               autoencoderX_loss, autoencoderY_loss,
                               feat_recon_loss])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_realX2Y=predict_realX2Y,
        predict_realY2X=predict_realY2X,
        predict_fakeX2Y=predict_fakeX2Y,
        predict_fakeY2X=predict_fakeY2X,
        im_swapped_X=im_swapped_X,
        im_swapped_Y=im_swapped_Y,
        sel_auto_X=sel_auto_X,
        sel_auto_Y=sel_auto_Y,
        discrimX2Y_loss=ema.average(discrimX2Y_loss),
        discrimY2X_loss=ema.average(discrimY2X_loss),
        discrimX2Y_grads_and_vars=discrimX2Y_grads_and_vars,
        discrimY2X_grads_and_vars=discrimY2X_grads_and_vars,
        genX2Y_loss_GAN=ema.average(genX2Y_loss_GAN),
        genY2X_loss_GAN=ema.average(genY2X_loss_GAN),
        genX2Y_loss_L1=ema.average(genX2Y_loss_L1),
        genY2X_loss_L1=ema.average(genY2X_loss_L1),
        genX2Y_grads_and_vars=genX2Y_grads_and_vars,
        genY2X_grads_and_vars=genY2X_grads_and_vars,
        outputsX2Y=outputsX2Y,
        outputsY2X=outputsY2X,
        outputsX2Yp=outputsX2Yp,
        outputsY2Xp=outputsY2Xp,
        auto_outputX = auto_outputX,
        autoencoderX_loss=autoencoderX_loss,
        autoencoderX_grads_and_vars=autoencoderX_grads_and_vars,
        auto_outputY = auto_outputY,
        autoencoderY_loss=autoencoderY_loss,
        autoencoderY_grads_and_vars=autoencoderY_grads_and_vars,
        feat_recon_loss=feat_recon_loss,
        feat_recon_grads_and_vars=feat_recon_grads_and_vars,
        train=tf.group(update_losses, incr_global_step, genX2Y_train,
                       genY2X_train, autoencoderX_train, autoencoderY_train,
                       feat_recon_train),
        train_disc = tf.group(discrimX2Y_train,discrimY2X_train)
    )



    
