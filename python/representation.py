#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import *
import os
from time import strftime
from robotic_priors import *

# Define constants
k = {
    "epsilon_a": 0.01,
    "epsilon_r": 0.01,
    "range_W": 0.05,
    "w_temp": 10,
    "w_prop": 1,
    "w_caus": 1,
    "w_rep": 5,
    "lambda": 0.0025,
    "dim_s": 2,
    "num_pair_samples": 100
}

if __name__ == "__main__":

    print(list_files())
    filename = get_filename(-2)
    print(filename)

    train_batch = batch_data(size_batch=10000, extra=True, filename=filename, dataset="train")
    test_batch  = batch_data(size_batch=10000, extra=True, filename=filename, dataset="test")
    o_test,  _, a_test, r_test, x_test,  _ = next(test_batch)

    # Model parameters
    img_width = 20
    img_height = 15
    img_depth = 3
    dim_o = img_width * img_height * img_depth
    dim_x = 2
    dim_a = 2
    dim_h1 = 4

    # Placeholders
    o = tf.placeholder(tf.float32, [None, dim_o], "o")
    a = tf.placeholder(tf.float32, [None, dim_a], "a")
    r = tf.placeholder(tf.float32, [None], "r")
    x = tf.placeholder(tf.float32, [None, dim_x], "x")
    feed_test = {
        o: o_test,
        x: x_test,
        a: a_test,
        r: r_test
    }

    # Model architecture
    initializer = tf.contrib.layers.xavier_initializer()
    W1 = tf.get_variable("W1", [dim_o, dim_h1], dtype=tf.float32, initializer=initializer)
    W2 = tf.get_variable("W2", [dim_h1, dim_x], dtype=tf.float32, initializer=initializer)
    tf.summary.tensor_summary("W1", W1)
    tf.summary.tensor_summary("W2", W2)
    # b = tf.Variable(tf.zeros([dim_y]), dtype=tf.float32)

    # Model input and output
    h1 = tf.nn.relu(tf.matmul(o, W1))
    s  = tf.matmul(h1, W2)
    # ds = s[1:] - s[:-1]
    # L_temp = tf.reduce_sum(tf.square(ds)) / tf.cast(tf.shape(ds)[0], tf.float32)

    L_temp = temporal_coherence_loss(s)
    L_prop = proportionality_loss(s, a)
    L_caus = causality_loss(s, a, r)
    L_rep  = repeatability_loss(s, a)

    # loss
    loss = L_caus
    # loss = tf.losses.mean_squared_error(x, s)
    tf.summary.scalar("loss", loss)

    # optimizer
    # optimizer = tf.train.AdadeltaOptimizer()
    # optimizer = tf.train.RMSPropOptimizer(0.0001)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    train = optimizer.minimize(loss)

    merged = tf.summary.merge_all()
    # datadir = os.path.join("data", "logs", strftime("%m-%d_%H-%M"))
    # train_writer = tf.summary.FileWriter(os.path.join(datadir, "train"))
    # test_writer = tf.summary.FileWriter(os.path.join(datadir, "test"))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(100):
        o_train, _, a_train, r_train, x_train, _ = next(train_batch)
        feed_train = {
            o: o_train,
            x: x_train,
            a: a_train,
            r: r_train
        }
        sess.run(train, feed_train)

        # evaluate training accuracy
        summary, loss_train = sess.run([merged, loss], feed_train)
        # train_writer.add_summary(summary, i)
        summary, loss_test  = sess.run([merged, loss], feed_test)
        # test_writer.add_summary(summary, i)
        print("Train loss: {}, Test loss: {}".format(loss_train, loss_test))
        # print(curr_y - y_train)
        # print(curr_W1)
