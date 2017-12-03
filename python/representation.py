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
    LOG = True

    print(list_files())
    filename = get_filename(-1)
    print(filename)

    # Model parameters
    dim_o = np.prod(np.array(get_observation_dim(filename)))
    dim_x = 2
    dim_a = 2
    dim_h1 = 4

    train_batch = batch_data(size_batch=100, extra=True, filename=filename, dataset="train")
    test_batch  = batch_data(size_batch=100, extra=True, filename=filename, dataset="test")
    o_test, a_test, r_test, x_test,  _ = next(test_batch)

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
    b1 = tf.get_variable("b1", [dim_h1], dtype=tf.float32, initializer=initializer)
    b2 = tf.get_variable("b2", [dim_x], dtype=tf.float32, initializer=initializer)

    # Model input and output
    h1 = tf.nn.relu(tf.matmul(o, W1) + b1)
    s  = tf.matmul(h1, W2) + b2

    L_temp = temporal_coherence_loss(s)
    L_prop = proportionality_loss(s, a)
    L_caus = causality_loss(s, a, r)
    L_rep  = repeatability_loss(s, a)

    # loss
    loss = k["w_temp"] * L_temp + k["w_prop"] * L_prop + k["w_caus"] * L_caus + k["w_rep"] * L_rep
    tf.summary.scalar("loss", loss)

    # optimizer
    optimizer = tf.train.AdadeltaOptimizer()
    # optimizer = tf.train.RMSPropOptimizer(0.0001)
    train = optimizer.minimize(loss)

    merged = tf.summary.merge_all()
    if LOG:
        datadir = os.path.join("data", "logs", strftime("%m-%d_%H-%M"))
        modeldir = os.path.join("data", "models", strftime("%m-%d_%H-%M"))
        modelpath = os.path.join(modeldir, "model")
        train_writer = tf.summary.FileWriter(os.path.join(datadir, "train"))
        test_writer = tf.summary.FileWriter(os.path.join(datadir, "test"))
        saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    if LOG:
        saver.save(sess, modelpath)
    for i in range(10):
        o_train, a_train, r_train, x_train, _ = next(train_batch)
        feed_train = {
            o: o_train,
            x: x_train,
            a: a_train,
            r: r_train
        }
        sess.run(train, feed_train)
        if LOG:
            saver.save(sess, modelpath, write_meta_graph=False)

        # evaluate training accuracy
        summary, loss_train = sess.run([merged, loss], feed_train)
        if LOG:
            train_writer.add_summary(summary, i)
        summary, loss_test  = sess.run([merged, loss], feed_test)
        if LOG:
            test_writer.add_summary(summary, i)

        print("Train loss: {}, Test loss: {}".format(loss_train, loss_test))
