#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 James James Johnson. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
@author: James J Johnson
@url: https://www.linkedin.com/in/james-james-johnson/
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # drop NUMA warnings from TF
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def generate_noisy_linear_dependency(
        beta0, beta1, sigma = 0.02, n = 500,
        seed_x = None, seed_eps = None) :

    '''
    Generate random noisy linear dependency.

    Args:
      beta0: shift of the scaled independent variable
      beta1: scaler of the independent variable
      sigma: (optional float) standard devaition of the residuals
      n: (optional int) the count of randomly generated examples
      seed_x: (optional int): random seed for the independent variable
      seed_eps: (optional int): random seed for the residuals

    Returns:
      Tuple of two 1D random tensors: independent and dependednt variables
    '''

    # Step 1: generate random samples for independent variable and residual
    x = tf.random.uniform(shape=(n,), seed=seed_x)
    epsilon = tf.random.normal(
        shape=(len(x),), stddev=sigma, seed=seed_eps)

    # Step 2: compute random samples for dependent variable
    y = beta1 * x + beta0 + epsilon

    return (x, y)


def predict(betas, x) :
    '''
    Predict independent variable examples from the dependent variable examples
    and model parameters

    Args:
      betas: 1D list of model paramaters of size 2.
      x: 1D tensor with examples of independednt variable

    Returns:
      1D tensor with examples of predicted values of dependednt variable
    '''
    y = betas[1] * x + betas[0]
    return y


def mean_squared_error(y_pred, y_true) :
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    return mse


def fit(x, y, betas_in_out, epochs, learning_rate, verbose=True) :

    history = []

    # For each epoch ...
    for epoch in range(epochs):

        # Inititalize gradient tape recoring for forward-prpagation stage
        with tf.GradientTape() as tape:

            # Step 1: Forward-propagation with predictions recording
            predictions = predict(betas = betas_in_out, x = x)

            # Step 2: Optimization target (loss) computation with recording
            loss = mean_squared_error(y_pred = predictions, y_true = y)

        # Step 3: Back-propagation with gradient comptation from recorded tape
        gradients = tape.gradient(target = loss, sources = betas_in_out)

        # Step 4: save series of evolving parameters and losses for reporting
        history.append((betas_in_out[0].numpy(),
                        betas_in_out[1].numpy(),
                        loss.numpy()))

        # Step 5. Update ALL parameters based on their respective gradients
        betas_in_out[0].assign_sub(gradients[0] * learning_rate)
        betas_in_out[1].assign_sub(gradients[1] * learning_rate)

        # Step 6: Print trace in real time diring fitting
        if verbose :
            if epoch % int(epochs / 10) == 0:
                print("Epoch {:d} Loss {:.10f}".format(epoch, loss.numpy()))

    return history


def plot_gradient_descent_path(history, x, y):
    beta0s = np.linspace(-1, 1)
    beta1s = np.linspace(-1, 1)
    beta0s_mesh, beta1s_mesh = np.meshgrid(beta0s, beta1s)
    losses = np.array([mean_squared_error(
        y_pred = predict(betas = [beta0, beta1], x = x),
        y_true = y)
        for(beta0, beta1) in
            zip(np.ravel(beta0s_mesh),
                np.ravel(beta1s_mesh))])
    loss_mesh = losses.reshape(beta0s_mesh.shape)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection = "3d")
    ax.plot_surface(
        beta0s_mesh,
        beta1s_mesh,
        loss_mesh,
        color='b',
        alpha=0.05)
    ax.plot([h[0]for h in history],
            [h[1]for h in history],
            [h[2]for h in history],
            marker='o')
    ax.set_xlabel('beta0', fontsize = 17, labelpad=10)
    ax.set_ylabel('beta1', fontsize = 17, labelpad=10)
    ax.set_zlabel('loss', fontsize = 17, labelpad=10)
    ax.view_init(elev = 20, azim=10)


BETA0_TRUE = +0.40
BETA1_TRUE = +0.15
RESIDUAL_STDEV = 0.02
NUM_RANDOM_SAMPLES = 500
INDEP_VAR_RAND_SEED = 1
RESID_VAR_RAND_SEED = 2

BETA0_INIT = -0.85
BETA1_INIT = -0.55
EPOCHS = 1001
LEARNING_RATE = 0.04


def main() :

    ###########################################################################
    # Generate and plot random training dataset
    print("Generating random training dataset... ", end = "")
    x_train, y_train = generate_noisy_linear_dependency(
        beta0 = BETA0_TRUE,
        beta1 = BETA1_TRUE,
        sigma = RESIDUAL_STDEV,
        n = NUM_RANDOM_SAMPLES,
        seed_x = INDEP_VAR_RAND_SEED,
        seed_eps = RESID_VAR_RAND_SEED,
        )
    print("Done.")
    print()

    ###########################################################################
    # Report CPU/GPU availability
    print("Fitting will be using {int_cpu_count:d} CPU(s).".format(
        int_cpu_count = len(tf.config.list_physical_devices('CPU'))))
    print("Fitting will be using {int_gpu_count:d} GPU(s).".format(
        int_gpu_count = len(tf.config.list_physical_devices('GPU'))))
    print()

    ###########################################################################
    # Initialize model parameters, and compute initial loss
    betas = [tf.Variable(BETA0_INIT), tf.Variable(BETA1_INIT)]
    loss = mean_squared_error(y_pred = predict(
        betas = betas, x = x_train), y_true=y_train)
    print("Initial loss = {init_loss:.10f}".format(init_loss=loss.numpy()))
    print()

    ###########################################################################
    # Fit the model
    print("Started model fitting ...")
    history = fit(x = x_train, y = y_train, betas_in_out = betas,
                epochs=EPOCHS, learning_rate = LEARNING_RATE, verbose=True)
    print("Finished model fitting.")
    print()

    ###########################################################################
    # Report estimated parameters.
    print("Estimation results:")
    print()
    print("beta0_true: {:.6f}".format(BETA0_TRUE))
    print("beta0_init: {:.6f}".format(BETA0_INIT))
    print("beta0_hat : {:.6f}".format(betas[0].numpy()))
    print()
    print("beta1_true: {:.6f}".format(BETA1_TRUE))
    print("beta1_init: {:.6f}".format(BETA1_INIT))
    print("beta1_hat : {:.6f}".format(betas[1].numpy()))
    print()

    ###########################################################################
    # Plot the dataset, and the fitted model.
    print("Producing scatter plot from dataset and predictions line... ",
          end = "")
    plt.plot(x_train, y_train, "b.")
    plt.plot(x_train, predict(betas = betas, x = x_train))
    print("Done.")
    print()

    ###########################################################################
    # Plot gradient descent path
    print("Producing 3D plot of gradient descent path... ",
          end = "")
    plot_gradient_descent_path(history=history, x=x_train, y=y_train)
    print("Done.")
    print()


if __name__ == "__main__" :
    main()
