import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import gym
import gym_sai2

import numpy as np
import numpy.random as rand
from data import *
from robotic_priors import *
from reinforcement_learning import *
from configs.dqn import config
import threading

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def action(self, observation):
        return self.action_space.sample()

class DQN_Agent(object):
    
    def __init__(self, env, sess, config, logdir, logger=None):

            
        # store hyper params
        self.config = config
        self.sess=tf.Session() #sess
        self.logdir=logdir
        #self.logger = logger #no idea how to use tboard
        self.env = env
        self.s_dim=2
        self.a_dim=2
        self.a_size=5

        # build model
        self.build()
        #make summary writer
        self.train_writer = tf.summary.FileWriter(self.logdir)

        self.initialize()

    def build(self):
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        self.q = self.get_q_values_op(self.s, scope="q", reuse=False)

        # compute Q values of next state
        self.target_q = self.get_q_values_op(self.sp, scope="target_q", reuse=False)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q, self.target_q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")

    def update_averages(self, rewards, q_values, scores_eval):
        self.avg_reward = np.mean(rewards)
        self.avg_q      = np.mean(q_values)
    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
        #self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss dqn", self.loss)
        tf.summary.scalar("grads norm dqn", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward DQN", self.avg_reward_placeholder)
        tf.summary.scalar("Avg Q DQN", self.avg_q_placeholder)
        #tf.summary.scalar("Eval Reward DQN", self.eval_reward_placeholder)
            
        # logging
        self.summary = tf.summary.merge_all()
        #self.file_writer = tf.summary.FileWriter(self.config.output_path, self.sess.graph)
    def initialize(self):
        self.add_summary()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(self.update_target_op) #sync target
        #self.saver = tf.train.Saver()
    
    def add_placeholders_op(self):
        self.s=tf.placeholder(tf.float32, shape=(None, self.s_dim), name='dqn_s')
        self.a=tf.placeholder(tf.int32, shape=(None), name='dqn_a')
        self.dqn_r=tf.placeholder(tf.float32, shape=(None), name='dqn_r')
        self.sp=tf.placeholder(tf.float32,  shape=(None, self.s_dim), name='dqn_sp')
        self.done_mask=tf.placeholder(tf.bool, shape=(None),  name='dqn_done')
        self.lr=tf.placeholder(tf.float32, name='dqn_lr')
        #elf.lr=.002

    def get_q_values_op(self, state, scope, reuse=False):
        #q arch here
        with tf.variable_scope(scope):
            #out = tf.layers.dense(layers.flatten(state), num_actions, activation=None, reuse=reuse, name='fc')
            hidden= layers.fully_connected(state, 10, activation_fn=tf.nn.relu, reuse=reuse)
            out= layers.fully_connected(hidden, self.a_size**self.a_dim, activation_fn=None, reuse=reuse)
        return out
    def action(self, state, eps):
        #action index to action in [-1, -.5, 0, .5, 1] x[-1, -.5, 0, .5, 1]
        if rand.random() > eps:
            action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
            actInd= np.argmax(action_values)
        else:
            actInd=rand.randint(self.a_size**self.a_dim) # rand on 0-4x0-4 U Z

        
        act=np.array(np.unravel_index(actInd, (self.a_size, self.a_size)))
        act=2.*act/float(self.a_size-1) -1
        return act, actInd
    
    def add_update_target_op(self, q_scope, target_q_scope):
        target_q_vals = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        q_vals = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        self.update_target_op=tf.group(*[tf.assign(tar_q, q) for tar_q, q in zip(target_q_vals, q_vals)])
    
    def add_loss_op(self, q, target_q):
        num_actions = self.a_size**self.a_dim
        q_samp=self.dqn_r +(self.config.gamma*tf.reduce_max(target_q, axis=1))*tf.cast(tf.logical_not(self.done_mask), tf.float32)
        action_mask=tf.cast(tf.one_hot(self.a, num_actions, 1, 0), tf.float32)
        self.loss=tf.reduce_mean(tf.square(q_samp-tf.reduce_sum(q * action_mask, axis=1)))

    def add_optimizer_op(self, scope):
        Adam_Optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        grads_and_vars = Adam_Optimizer.compute_gradients(self.loss, var_list)
        if self.config.grad_clip:
            processed_gv=[(tf.clip_by_norm(grad, self.config.clip_val), val) for grad, val in grads_and_vars]
        else:
            processed_gv=grads_and_vars
        grads = [gv[0] for gv in processed_gv]
        self.train_op = Adam_Optimizer.apply_gradients(processed_gv)
        self.grad_norm = tf.global_norm(grads)

    def train_step(self, data, lr):
        #Performs an update of parameters with the given data


        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = data
        print('training a step')
        print(r_batch)

        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.dqn_r: r_batch,
            self.sp: sp_batch, 
            self.done_mask: done_mask_batch,
            self.lr: lr, 
            # extra info
            self.avg_reward_placeholder: 0, #self.avg_reward,  
            self.avg_q_placeholder: 0 #self.avg_q
            }

        loss, grad, summary, _ = self.sess.run([self.loss, self.grad_norm, self.summary, self.train_op], feed_dict=fd)


        return loss, grad, summary

    def train_network(self, train_batch, lr):
        min_loss_train = float("inf")
        print("TRAINING: DQN: lr={},".format(lr))

        i = 0
        for o_train, a_train, r_train, x_train, dx_train, s_hat_train, ai_train, sp_hat_train, d_train in train_batch:
            # Train iteration
            dataForRl=(s_hat_train, ai_train, r_train, sp_hat_train, d_train)
            loss_train, grad_train, summary_train = self.train_step(dataForRl, lr)

            if i%10==0:
                self.sess.run(self.update_target_op)


            self.train_writer.add_summary(summary_train, i)

            # if loss_train < min_loss_train:
            #     min_loss_train = loss_train
            #     self.saver.save(self.sess, self.modelpath, global_step=i)
            #     with open(os.path.join(self.modeldir, "saved.log"), "w+") as f:
            #         f.write("Iteration: {}, Train loss: {}".format(i, loss_train))

            print("Iteration: {}, DQN Train loss: {},".format(i, loss_train))
            i += 1

# class REINFORCE_Agent(object, sess):
    
#     def __init__(self, env, sess, config, logger=None):

#         if not os.path.exists(config.output_path):
#             os.makedirs(config.output_path)
            
#         # store hyper params
#         self.initialize(sess)
#         self.config = config
#         self.logger = logger
#         if logger is None:
#             self.logger = get_logger(config.log_path)
#         self.env = env
#         self.s_dim=2
#         self.a_dim=2
#         self.a_size=5

#         # build model
#         self.build()

#     def build(self):
#         # add placeholders
#         self.add_placeholders_op()

#         # compute Q values of state
#         s = self.process_state(self.s)
#         self.q = self.get_q_values_op(s, scope="q", reuse=False)

#         # compute Q values of next state
#         sp = self.process_state(self.sp)
#         self.target_q = self.get_q_values_op(sp, scope="target_q", reuse=False)

#         # add update operator for target network
#         self.add_update_target_op("q", "target_q")

#         # add square loss
#         self.add_loss_op(self.q, self.target_q)

#         # add optmizer for the main networks
#         self.add_optimizer_op("q")

#         self.sess.run(self.update_target_op)
#     def initialize(self, sess):
#         self.sess = sess
#         #self.add_summary()
#         init = tf.global_variables_initializer()
#         self.sess.run(init)
#         self.sess.run(self.update_target_op) #sync target
#         #self.saver = tf.train.Saver()


if __name__ == "__main__":
    env = gym.make("sai2-v0")
    print(env.observation_space.shape)
