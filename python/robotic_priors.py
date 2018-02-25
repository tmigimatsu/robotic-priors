"""
Tensorflow ops for robotic prior losses.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

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

        # TODO: Should be T+1 x 2
        grad = 2 / T * diff_ds

        return loss, grad

    def _TemporalCoherenceLossGrad(op, grad_loss, grad_grad):
        return grad_loss * op.outputs[1]

    with ops.name_scope(name, "TemporalCoherenceLoss", [s]) as name:
        loss, _ = py_func(_temporal_coherence_loss_grad,
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
        if idx_a1.shape[0] == 0:
            return np.float32(0), np.zeros(s.shape, dtype=np.float32)

        # Loss
        ds = s[1:] - s[:-1]
        ds_a1 = ds[idx_a1]
        ds_a2 = ds[idx_a2]

        ds_a1_norm = np.float32(np.linalg.norm(ds_a1, axis=1))
        ds_a2_norm = np.float32(np.linalg.norm(ds_a2, axis=1))

        # Prevent divide by 0
        idx = np.logical_and(ds_a1_norm != 0, ds_a2_norm != 0)
        if np.any(idx == 0):
            idx_a1 = idx_a1[idx]
            idx_a2 = idx_a2[idx]
            ds_a1 = ds[idx_a1]
            ds_a2 = ds[idx_a2]
            ds_a1_norm = np.float32(np.linalg.norm(ds_a1, axis=1))
            ds_a2_norm = np.float32(np.linalg.norm(ds_a2, axis=1))

        dds_norm = ds_a1_norm - ds_a2_norm

        loss = np.float32(1 / T) * dds_norm.dot(dds_norm)

        # Gradient
        I = np.eye(T+1, dtype=np.float32)
        D = I[1:] - I[:-1]
        diff_a1 = D[idx_a1]
        diff_a2 = D[idx_a2]

        q1 = diff_a1.T.dot((dds_norm / ds_a1_norm)[:,np.newaxis] * ds_a1)
        q2 = diff_a2.T.dot((dds_norm / ds_a2_norm)[:,np.newaxis] * ds_a2)

        grad = 2 / T * (q1 - q2)

        return loss, grad

    def _ProportionalityLossGrad(op, grad_loss, grad_grad):
        return grad_loss * op.outputs[1], 0 * op.inputs[1]

    with ops.name_scope(name, "ProportionalityLoss", [s, a]) as name:
        loss, _ = py_func(_proportionality_loss_grad,
                          [s, a], [tf.float32, tf.float32], name=name,
                          grad=_ProportionalityLossGrad)
        return loss

def sample_action_reward_predicates(a, r):
    idx_ar1 = []
    idx_ar2 = []
    for i in range(a.shape[0]-1):
        idx_i_a  = np.linalg.norm(a[i,np.newaxis,:] - a[i+1:,:], axis=1) < k["epsilon_a"]
        idx_i_r  = np.abs(r[i] - r[i+1:]) > k["epsilon_r"]
        idx_i_ar = np.logical_and(idx_i_a, idx_i_r)
        idx_i_ar = np.random.permutation(np.where(idx_i_ar)[0])
        num_ar   = min(k["num_pair_samples"], len(idx_i_ar))
        idx_i_ar = idx_i_ar[:num_ar] + i+1

        idx_ar1.append(i + np.zeros(num_ar, dtype=np.int32))
        idx_ar2.append(idx_i_ar)

    idx_ar1 = np.concatenate(idx_ar1)
    idx_ar2 = np.concatenate(idx_ar2)

    return (idx_ar1, idx_ar2)

def causality_loss(s, a, r, name=None):
    def _causality_loss_grad(s, a, r):
        T = s.shape[0] - 1
        S = s.shape[1]
        idx_ar1, idx_ar2 = sample_action_reward_predicates(a, r)
        if idx_ar1.shape[0] == 0:
            return np.float32(0), np.zeros(s.shape, dtype=np.float32)

        # Loss
        s = s[1:]
        s_ar1 = s[idx_ar1]
        s_ar2 = s[idx_ar2]
        ds = s_ar1 - s_ar2
        ds_norm = (ds * ds).sum(axis=1)

        c = ds_norm.min()
        exp_ds_norm = np.exp(c - ds_norm, dtype=np.float32)

        loss = np.float32(np.exp(-c) / T) * exp_ds_norm.sum()

        # Gradient
        I = np.eye(T+1, dtype=np.float32)
        d_ar = I[idx_ar1] - I[idx_ar2]
        d_ar_exp_ds_norm_ds = d_ar.T.dot(exp_ds_norm[:,np.newaxis] * ds)

        # TODO: Should be T+1 x 2
        grad = -2 * np.exp(-c) / T * d_ar_exp_ds_norm_ds

        return loss, grad

    def _CausalityLossGrad(op, grad_loss, grad_grad):
        return grad_loss * op.outputs[1], 0 * op.inputs[1], 0 * op.inputs[2]

    with ops.name_scope(name, "CausalityLoss", [s, a, r]) as name:
        loss, _ = py_func(_causality_loss_grad,
                          [s, a, r], [tf.float32, tf.float32], name=name,
                          grad=_CausalityLossGrad)
        return loss

def repeatability_loss(s, a, name=None):
    def _repeatability_loss_grad(s, a):
        T = s.shape[0] - 1
        S = s.shape[1]
        idx_a1, idx_a2 = sample_action_predicates(a)
        if idx_a1.shape[0] == 0:
            return np.float32(0), np.zeros(s.shape, dtype=np.float32)

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

        loss = np.float32(np.exp(-c) / T) * exp_ds_dds_norm.sum()

        # Gradient
        # TODO: Find more efficient way to find A1 - A2
        I = np.eye(T, dtype=np.float32)
        da = I[idx_a1] - I[idx_a2]
        I = np.eye(T+1, dtype=np.float32)
        D  = I[1:] - I[:-1]
        diff_da = da.dot(D)

        X = diff_da.T.dot(exp_ds_norm[:,np.newaxis] * dds)
        Y = da.T.dot(exp_ds_dds_norm[:,np.newaxis] * ds)
        Y = np.vstack((np.zeros(Y.shape[1], dtype=np.float32), Y))

        grad = np.float32(2 * np.exp(-c) / T) * (X - Y)

        return loss, grad

    def _RepeatabilityLossGrad(op, grad_loss, grad_grad):
        return grad_loss * op.outputs[1], 0 * op.inputs[1]

    with ops.name_scope(name, "RepeatabilityLoss", [s, a]) as name:
        loss, _ = py_func(_repeatability_loss_grad,
                             [s, a], [tf.float32, tf.float32], name=name,
                             grad=_RepeatabilityLossGrad)
        return loss

