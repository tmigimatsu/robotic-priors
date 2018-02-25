"""
Variational autoencoder (experimental).
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import *
import os
from time import strftime
from robotic_priors import *

print(list_files())
filename = get_filename(-1)
print(filename)

train_batch = batch_data(size_batch=1000, extra=True, filename=filename, dataset="train", reshape=False)
test_batch  = batch_data(size_batch=5000, extra=True, filename=filename, dataset="test", reshape=False)
x_test, _, _, _, _ = next(test_batch)

dim_image = list(get_observation_dim(filename))
dim_o = 2*28224#np.prod(dim_image)
dim_x = 2
dim_a = 2
dim_h1 = 32
dim_h2 = 8
krn_h1 = 5
krn_h2 = 5
input_dim = np.prod(dim_image)#np.prod(np.array(get_observation_dim(filename)))
hidden_encoder_dim = 400
hidden_decoder_dim = 1024
latent_dim = 2
lam = 0

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

x = tf.placeholder("float", shape=[None] + dim_image, name="x")
x_flat = tf.reshape(x, [-1, input_dim])
l2_loss = tf.constant(0.0)

conv1 = tf.layers.conv2d(
    inputs=x,
    filters=dim_h1,
    kernel_size=(krn_h1, krn_h1),
    padding="valid",
    strides=(2,2),
    activation=tf.nn.relu,
    name="conv1"
)
# 98 x 73 x 16
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=(2,2),
    strides=2,
    name="pool1"
)
# 49 x 36 x 16
h1 = tf.reshape(pool1, [-1, dim_o])

W_encoder_input_hidden = weight_variable([dim_o,hidden_encoder_dim])
b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

# Hidden layer encoder
# hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)
hidden_encoder = tf.nn.relu(tf.matmul(h1, W_encoder_input_hidden) + b_encoder_input_hidden)

W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_mu = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

# Mu encoder
mu_encoder = tf.add(tf.matmul(hidden_encoder, W_encoder_hidden_mu), b_encoder_hidden_mu, name= "mu_encoder")

W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_logvar = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

# Sigma encoder
logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable
std_encoder = tf.exp(0.5 * logvar_encoder)
z = mu_encoder + tf.multiply(std_encoder, epsilon)

W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
b_decoder_hidden_reconstruction = bias_variable([input_dim])
l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)

x_hat = tf.add(tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction), b_decoder_hidden_reconstruction, name="x_hat")
BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x_flat), reduction_indices=1)

loss = tf.reduce_mean(BCE + KLD)

regularized_loss = loss + lam * l2_loss

loss_sum = tf.summary.scalar("lowerbound", loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.summary.merge_all()

# add Saver ops
saver = tf.train.Saver()

n_steps = int(1e6)
batch_size = 100

datadir = os.path.join("data", "logs", strftime("%m-%d_%H-%M"))
modeldir = os.path.join("data", "models", strftime("%m-%d_%H-%M"))
modelpath = os.path.join(modeldir, "model")
train_writer = tf.summary.FileWriter(os.path.join(datadir, "train"))
test_writer = tf.summary.FileWriter(os.path.join(datadir, "test"))
saver = tf.train.Saver()
min_loss_test = float("inf")

with tf.Session() as sess:
  summary_writer = tf.summary.FileWriter('experiment',
                                          graph=sess.graph)
  if os.path.isfile("save/model.ckpt"):
    print("Restoring saved parameters")
    saver.restore(sess, "save/model.ckpt")
  else:
    print("Initializing parameters")
    sess.run(tf.global_variables_initializer())

  for step in range(1000):
    x_train, _, _, _, _ = next(train_batch)
    _, loss_train, summary_train = sess.run([train_step, loss, summary_op], feed_dict={x: x_train})
    loss_test, summary_test = sess.run([loss, summary_op], feed_dict={x: x_test})

    if step % 100 == 0:
      train_writer.add_summary(summary_train, step)
      test_writer.add_summary(summary_test, step)
    if loss_test < min_loss_test:
      min_loss_test = loss_test
      save_path = saver.save(sess, modelpath, global_step=step)
    print("Step {0} | Loss: {1}, {2}".format(step, loss_train, loss_test))

  train_writer.close()
  test_writer.close()
