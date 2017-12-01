#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import *
import os
from time import strftime
from tensorflow.python.framework import ops

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

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def temporal_coherence_loss(s, name=None):
    def _temporal_coherence_loss_grad(s):
        T = s.shape[0] - 1

        # Loss
        ds = s[1:] - s[:-1]

        loss = np.float32(1 / T) * (ds * ds).flatten().sum()

        # Gradient
        diff_ds = ds[:-1] - ds[1:]
        diff_ds = np.vstack((-ds[0], diff_ds, ds[-1]))

        grad = 2 / T * diff_ds

        return loss, grad

    def _TemporalCoherenceLossGrad(op, grad_loss, grad_grad):
        return grad_loss * op.outputs[1]

    with ops.name_scope(name, "TemporalCoherenceLoss", [s]) as name:
        loss, grad = py_func(_temporal_coherence_loss_grad,
                             [s], [tf.float32, tf.float32], name=name,
                             grad=_TemporalCoherenceLossGrad)
        return loss

def sample_action_predicates(a):
    idx_a1 = []
    idx_a2 = []
    for i in range(a.shape[0]-1):
        idx_i_a  = np.linalg.norm(a[i,np.newaxis,:] - a[i+1:,:], axis=1) < k["epsilon_a"]
        idx_i_a  = np.random.permutation(np.where(idx_i_a)[0])
        num_a    = min(k["num_pair_samples"], len(idx_i_a))
        idx_i_a  = idx_i_a[:num_a] + i+1

        idx_a1.append(i + np.zeros(num_a, dtype=np.int32))
        idx_a2.append(idx_i_a)

    idx_a1 = np.concatenate(idx_a1)
    idx_a2 = np.concatenate(idx_a2)

    return (idx_a1, idx_a2)

def proportionality_loss(s, a, name=None):
    def _proportionality_loss_grad(s, a):
        T = s.shape[0] - 1
        S = s.shape[1]
        idx_a1, idx_a2 = sample_action_predicates(a)

        # Loss
        ds = s[1:] - s[:-1]
        ds_a1 = ds[idx_a1]
        ds_a2 = ds[idx_a2]

        ds_a1_norm = np.linalg.norm(ds_a1, axis=1)
        ds_a2_norm = np.linalg.norm(ds_a2, axis=1)
        dds_norm = ds_a1_norm - ds_a2_norm

        loss = 1 / T * dds_norm.dot(dds_norm)

        # Gradient
        D = np.eye(T+1)[1:] - np.eye(T+1)[:-1]
        diff_a1 = D[idx_a1]
        diff_a2 = D[idx_a2]

        diff_ds_a1 = diff_a1.T.dot(ds_a1)
        diff_ds_a2 = diff_a2.T.dot(ds_a2)

        print(ds_a1_norm)
        print(ds_a2_norm)

        q1 = dds_norm.dot(1 / ds_a1_norm)
        q2 = dds_norm.dot(1 / ds_a2_norm)
        print("HELLO6")

        grad = 2 / T * (q1 * diff_ds_a1 - q2 * diff_ds_a2)
        print("HELLO7")

        return loss, grad

    def _ProportionalityLossGrad(op, grad_loss, grad_grad):
        return grad_loss * op.outputs[1], 0 * op.inputs[1]

    with ops.name_scope(name, "ProportionalityLoss", [s, a]) as name:
        loss, grad = py_func(_proportionality_loss_grad,
                             [s, a], [tf.float32, tf.float32], name=name,
                             grad=_ProportionalityLossGrad)
        return loss

def sample_action_reward_predicates(a, r):
    idx_ar1 = []
    idx_ar2 = []
    for i in range(a.shape[0]-1):
        idx_i_a  = np.linalg.norm(a[i,np.newaxis,:] - a[i+1:,:], axis=1) < k["epsilon_a"]
        idx_i_r  = np.linalg.norm(r[i,np.newaxis,:] - r[i+1:,:], axis=1) > k["epsilon_r"]
        idx_i_ar = np.logical_and(idx_i_a, idx_i_r)
        idx_i_ar = np.random.permutation(np.where(idx_i_ar)[0])
        num_ar   = min(k["num_pair_samples"], len(idx_i_ar))
        idx_i_ar = idx_i_ar[:num_ar] + i+1

        idx_ar1.append(i + np.zeros(num_ar, dtype=np.int32))
        idx_ar2.append(idx_i_a)

    idx_ar1 = np.concatenate(idx_ar1)
    idx_ar2 = np.concatenate(idx_ar2)

    return (idx_ar1, idx_ar2)

def causality_loss(s, a, r, name=None):
    def _causality_loss_grad(s, a, r):
        T = s.shape[0] - 1
        S = s.shape[1]
        idx_ar1, idx_ar2 = sample_action_reward_predicates(a, r)

        # Loss
        s = s[1:]
        s_ar1 = s[idx_ar1]
        s_ar2 = s[idx_ar2]
        ds = s_ar1 - s_ar2
        ds_norm = (ds * ds).sum(axis=1)
        c = ds_norm.min()
        exp_ds_norm = np.exp(c - ds_norm)

        loss = np.exp(-c) / T * exp_ds_norm.sum()

        # Gradient
        s = s[1:]
        s_ar1 = s[idx_ar1]
        s_ar2 = s[idx_ar2]
        ds = s_ar1 - s_ar2
        ds_norm = (ds * ds).sum(axis=1)
        
        c = ds_norm.min()
        exp_ds_norm = np.exp(c - ds_norm)

        loss = np.exp(-c) / T * exp_ds_norm.sum()

        # Gradient
        d_ar = np.eye(S)[idx_ar1] - np.eye(S)[idx_ar2]
        d_ar_exp_ds_norm_ds = d_ar.T.dot(exp_ds_norm[:,np.newaxis] * ds)

        grad = -2 * np.exp(-c) / T * d_ar_exp_ds_norm_ds

        return loss, grad

    def _CausalityLossGrad(op, grad_loss, grad_grad):
        return grad_loss * op.outputs[1]

    with ops.name_scope(name, "CausalityLoss", [s]) as name:
        loss, grad = py_func(_causality_loss_grad,
                             [s, a, r], [tf.float32, tf.float32, tf.float32], name=name,
                             grad=_CausalityLossGrad)
        return loss

def repeatability_loss(s, a, name=None):
    def _repeatability_loss_grad(s, a):
        T = s.shape[0] - 1
        S = s.shape[1]
        idx_a1, idx_a2 = sample_action_predicates(a)

        # Loss
        ds = s[1:] - s[:-1]
        s  = s[1:]
        s_a1  = s[idx_a1]
        s_a2  = s[idx_a2]
        ds_a1 = ds[idx_a1]
        ds_a2 = ds[idx_a2]
        ds  = s_a1 - s_a2
        dds = ds_a1 - ds_a2
        ds_norm  = (ds * ds).sum(axis=1)
        dds_norm = (dds * dds).sum(axis=1)
        c = ds_norm.min()
        exp_ds_norm = np.exp(c - ds_norm)
        exp_ds_dds_norm = exp_ds_norm * dds_norm

        loss = np.exp(-c) * exp_ds_dds_norm.sum() / T

        # Gradient
        da = np.eye(S)[idx_a1] - np.eye(S)[idx_a2]
        D  = np.eye(S)[1:] - np.eye(S)[:-1]
        diff_da = D.dot(da)

        X = diff_da.T.dot(exp_ds_norm[:,np.newaxis] * dds)
        Y = da.T.dot(exp_ds_dds_norm[:,np.newaxis] * ds)

        grad = 2 * np.exp(-c) / T * (X - Y)

        return loss, grad

    def _RepeatabilityLossGrad(op, grad_loss, grad_grad):
        return grad_loss * op.outputs[1]

    with ops.name_scope(name, "RepeatabilityLoss", [s]) as name:
        loss, grad = py_func(_repeatability_loss_grad,
                             [s, a], [tf.float32, tf.float32], name=name,
                             grad=_RepeatabilityLossGrad)
        return loss

if __name__ == "__main__":

    print(list_files())
    filename = get_filename(-2)
    print(filename)

    train_batch = batch_data(size_batch=10000, extra=True, filename=filename, dataset="train")
    test_batch  = batch_data(size_batch=10000, extra=True, filename=filename, dataset="test")
    o_test,  _, _, _, x_test,  _ = next(test_batch)

    # Model parameters
    img_width = 20
    img_height = 15
    img_depth = 3
    dim_o = img_width * img_height * img_depth
    dim_x = 2
    dim_a = 2
    dim_h1 = 4

    initializer = tf.contrib.layers.xavier_initializer()
    W1 = tf.get_variable("W1", [dim_o, dim_h1], dtype=tf.float32, initializer=initializer)
    tf.summary.tensor_summary("W1", W1)
    W2 = tf.get_variable("W2", [dim_h1, dim_x], dtype=tf.float32, initializer=initializer)
    tf.summary.tensor_summary("W2", W2)
    # b = tf.Variable(tf.zeros([dim_y]), dtype=tf.float32)

    # Model input and output
    o = tf.placeholder(tf.float32, [None, dim_o])
    h1 = tf.nn.relu(tf.matmul(o, W1))
    s  = tf.matmul(h1, W2)
    # ds = s[1:] - s[:-1]
    # L_temp = tf.reduce_sum(tf.square(ds)) / tf.cast(tf.shape(ds)[0], tf.float32)
    a = tf.placeholder(tf.float32, [None, dim_a])

    L_temp = temporal_coherence_loss(s)
    L_prop = proportionality_loss(s, a)

    x = tf.placeholder(tf.float32, [None, dim_x])

    # loss
    loss = L_prop
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
        o_train, _, a_train, _, x_train, _ = next(train_batch)
        sess.run(train, {o: o_train, x: x_train, a: a_train})

        # evaluate training accuracy
        summary, loss_train = sess.run([merged, loss], {o: o_train, x: x_train, a: a_train})
        # train_writer.add_summary(summary, i)
        summarx, loss_test  = sess.run([merged, loss], {o: o_test, x: x_test})
        # test_writer.add_summary(summary, i)
        print("Train loss: {}, Test loss: {}".format(loss_train, loss_test))
        # print(curr_y - y_train)
        # print(curr_W1)
