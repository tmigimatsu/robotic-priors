#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import *
import os
from time import strftime
from robotic_priors import *

# Define robotic prior constants
k = {
    "epsilon_a": 0.01,
    "epsilon_r": 0.01,
    "range_W": 0.05,
    "w_temp": 5,
    "w_prop": 1,
    "w_caus": 5,
    "w_rep": 5,
    "lambda": 0.0025,
    "dim_s": 2,
    "num_pair_samples": 1000
}

if __name__ == "__main__":
    LOG = True

    print(list_files())
    filename = get_filename(-1)
    print(filename)

    # Model parameters
    dim_image = np.array(get_observation_dim(filename))
    dim_o = np.prod(dim_image)
    dim_x = 2
    dim_a = 2
    dim_h1 = 16
    dim_h2 = 8
    krn_h1 = 5
    krn_h2 = 5

    # Initialize train and test datasets
    train_batch = batch_data(size_batch=1000, extra=True, filename=filename, dataset="train", flatten=True)
    test_batch  = batch_data(size_batch=1000, extra=True, filename=filename, dataset="test", flatten=True)
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
    # W1 = tf.get_variable("W1", [dim_o, dim_h1], dtype=tf.float32, initializer=initializer)
    # W2 = tf.get_variable("W2", [dim_h1, dim_x], dtype=tf.float32, initializer=initializer)
    # b1 = tf.get_variable("b1", [dim_h1], dtype=tf.float32, initializer=initializer)
    # b2 = tf.get_variable("b2", [dim_x], dtype=tf.float32, initializer=initializer)
    W0 = tf.get_variable("W0", [dim_o, dim_x], dtype=tf.float32, initializer=initializer)
    b0 = tf.get_variable("b0", [dim_o], dtype=tf.float32, initializer=initializer)
    # # 200 x 150 x 3
    # conv1 = tf.layers.conv2d(
    #     inputs=o,
    #     filters=dim_h1,
    #     kernel_size=(krn_h1, krn_h1),
    #     padding="valid",
    #     strides=(2,2),
    #     activation=tf.nn.relu,
    #     name="conv1"
    # )
    # # 98 x 73 x 16
    # pool1 = tf.layers.max_pooling2d(
    #     inputs=conv1,
    #     pool_size=(2,2),
    #     strides=2,
    #     name="pool1"
    # )
    # # 49 x 37 x 16
    # h1 = tf.reshape(pool1, [-1, dim_o])

    # Model input and output
    # h1 = tf.nn.relu(tf.matmul(o, W1) + b1)
    # s  = tf.add(tf.matmul(h1, W2), b2, name="s")
    s = tf.matmul(o + b0, W0, name="s")
    # s = tf.add(tf.matmul(h1, W0), b0, name="s")

    # Ground truth loss
    # loss = tf.losses.mean_squared_error(x, s)

    # Robotic rpiors loss
    L_temp = temporal_coherence_loss(s)
    L_prop = proportionality_loss(s, a)
    L_caus = causality_loss(s, a, r)
    L_rep  = repeatability_loss(s, a)
    loss = k["w_temp"] * L_temp + k["w_prop"] * L_prop + k["w_caus"] * L_caus + k["w_rep"] * L_rep

    tf.summary.scalar("loss", loss)

    # Create optimizer
    # optimizer = tf.train.AdadeltaOptimizer()
    # optimizer = tf.train.RMSPropOptimizer(0.0001)
    optimizer = tf.train.AdagradOptimizer(0.001)
    train = optimizer.minimize(loss)

    # Create logger
    merged = tf.summary.merge_all()
    if LOG:
        datadir = os.path.join("data", "logs", strftime("%m-%d_%H-%M"))
        modeldir = os.path.join("data", "models", strftime("%m-%d_%H-%M"))
        modelpath = os.path.join(modeldir, "model")
        train_writer = tf.summary.FileWriter(os.path.join(datadir, "train"))
        test_writer = tf.summary.FileWriter(os.path.join(datadir, "test"))
        saver = tf.train.Saver()
        min_loss_test = float("inf")

    # Initialize session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(10000):
        # Train iteration
        o_train, a_train, r_train, x_train, _ = next(train_batch)
        feed_train = {
            o: o_train,
            x: x_train,
            a: a_train,
            r: r_train
        }
        sess.run(train, feed_train)

        # Evaluate training accuracy
        summary_train, loss_train = sess.run([merged, loss], feed_train)
        summary_test, loss_test  = sess.run([merged, loss], feed_test)

        # Log summary
        if LOG and i % 100 == 0:
            train_writer.add_summary(summary_train, i)
            test_writer.add_summary(summary_test, i)
        if LOG:
            if loss_test < min_loss_test:
                min_loss_test = loss_test
                saver.save(sess, modelpath, global_step=i)
                with open(os.path.join(modeldir, "saved.log"), "w+") as f:
                    f.write("Iteration: {}, Train loss: {}, Test loss: {}".format(i, loss_train, loss_test))

        print("Iteration: {}, Train loss: {}, Test loss: {}".format(i, loss_train, loss_test))

    train_writer.close()
    test_writer.close()
