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
        self.a_size=7

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
        tf.summary.scalar("loss dqn", self.dqn_loss)
        tf.summary.scalar("grads norm dqn", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward DQN", self.avg_reward_placeholder)
        tf.summary.scalar("Avg Q DQN", self.avg_q_placeholder)
        #tf.summary.scalar("Eval Reward DQN", self.eval_reward_placeholder)
            
        # logging
        self.dqn_summary = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
    def initialize(self):
        self.add_summary()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(self.update_target_op) #sync target
        #self.saver = tf.train.Saver()
    
    def add_placeholders_op(self):
        self.s=tf.placeholder(tf.float32, shape=(None, self.s_dim), name='dqn_s')
        self.dqn_a=tf.placeholder(tf.int32, shape=(None), name='dqn_a')
        self.dqn_r=tf.placeholder(tf.float32, shape=(None), name='dqn_r')
        self.sp=tf.placeholder(tf.float32,  shape=(None, self.s_dim), name='dqn_sp')
        self.done_mask=tf.placeholder(tf.bool, shape=(None),  name='dqn_done')
        self.lr=tf.placeholder(tf.float32, name='dqn_lr')
        #elf.lr=.002

    def get_q_values_op(self, state, scope, reuse=False):
        #q arch here
        with tf.variable_scope(scope):
            #hidden= layers.fully_connected(layers.flatten(state), 10, activation_fn=tf.nn.relu, reuse=reuse)
            #out= layers.fully_connected(hidden, self.a_size**self.a_dim, activation_fn=None, reuse=reuse)
            out= layers.fully_connected(layers.flatten(state), self.a_size**self.a_dim, activation_fn=None, reuse=reuse)
        return out
    def action(self, state, eps):
        #action index to action in [-1, -.5, 0, .5, 1] x[-1, -.5, 0, .5, 1]
        if rand.random() > eps:
            #print(state[0])
            action_values = self.sess.run(self.q, feed_dict={self.s: state})[0]
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
        action_mask=tf.cast(tf.one_hot(self.dqn_a, num_actions, 1, 0), tf.float32)
        #self.debug_op=tf.reduce_sum(q * action_mask, axis=3)
        self.dqn_loss=tf.reduce_mean(tf.square(q_samp-tf.squeeze(tf.reduce_sum(q * action_mask, axis=3))))

    def add_optimizer_op(self, scope):
        Adam_Optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        grads_and_vars = Adam_Optimizer.compute_gradients(self.dqn_loss, var_list)
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
        #print('training a step')
        #print(r_batch)
        #print(self.sess.run(self.dqn_r, feed_dict={self.dqn_r: r_batch,}))
        fd = {
            # inputs
            self.s: s_batch,
            self.dqn_a: a_batch,
            self.dqn_r: r_batch,
            self.sp: sp_batch, 
            self.done_mask: done_mask_batch,
            self.lr: lr, 
            # extra info
            self.avg_reward_placeholder: 0, #self.avg_reward,  
            self.avg_q_placeholder: 0 #self.avg_q
            }
        #debuging stuff
        #G=self.sess.run(self.debug_op, feed_dict=fd)
        #print(G.shape)

        loss, grad, _ = self.sess.run([self.dqn_loss, self.grad_norm, self.train_op], feed_dict=fd)


        return loss, grad

    def train_network(self, train_batch, lr):
        min_loss_train = float("inf")
        print("TRAINING: DQN: lr={},".format(lr))

        i = 0
        for o_train, a_train, r_train, x_train, dx_train, s_hat_train, ai_train, sp_hat_train, d_train in train_batch:
            # Train iteration
            dataForRl=(s_hat_train, ai_train, r_train, sp_hat_train, d_train)
            loss_train, grad_train = self.train_step(dataForRl, lr)

            if i%10==0:
                self.sess.run(self.update_target_op)


            #self.train_writer.add_summary(summary_train, i)

            # if loss_train < min_loss_train:
            #     min_loss_train = loss_train
            #     self.saver.save(self.sess, self.modelpath, global_step=i)
            #     with open(os.path.join(self.modeldir, "saved.log"), "w+") as f:
            #         f.write("Iteration: {}, Train loss: {}".format(i, loss_train))

            print("\tIteration: {}, DQN Train loss: {},".format(i, loss_train))
            i += 1


class Reinforce_Agent(object):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """
  def __init__(self, env, config, dataLogDir,logger=None):
    """
    Initialize Policy Gradient Class
    """
    self.output_path=dataLogDir
    if not os.path.exists(self.output_path):
      os.makedirs(self.output_path)
            
    # store hyper-params
    self.config = config
    # self.logger = logger
    self.sess=tf.Session()
    # if logger is None:
    #   self.logger = get_logger(config.log_path)
    self.env = env
  
    # discrete action space or continuous action space
    self.discrete = True
    self.observation_dim = 2
    self.action_dim = 2
    
  
    self.lr = self.config.learning_rate
  
    # build model
    self.build()

  def build_mlp(self, mlp_input, output_size, scope, n_layers, size, output_activation=None):
    print(scope)
    with tf.variable_scope(scope):
      stackz=layers.fully_connected(inputs=mlp_input, num_outputs=size, activation_fn=tf.nn.relu)
      for i in range(n_layers-1):
        stackz=layers.fully_connected(inputs=stackz,  num_outputs=size, activation_fn=tf.nn.relu)
      out= layers.fully_connected(inputs=stackz,  num_outputs=output_size, activation_fn=output_activation)
    return out 
  
  def add_placeholders_op(self):
    self.observation_placeholder =tf.placeholder(tf.float32, shape=(None,self.observation_dim), name="obs") 
    #continous action distribution s
    self.action_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim), name='action')  
  
    # Define a placeholder for advantages
    self.advantage_placeholder = tf.placeholder(tf.float32, shape=(None), name='adv' )
  
  def build_policy_network_op(self, scope = "policy_network"):
    
    action_means =self.build_mlp(self.observation_placeholder, self.action_dim, scope,
                    self.config.n_layers,
                    self.config.layer_size) 

    log_std = tf.get_variable(shape=self.action_dim, name='log_std') # TODO 
    self.sampled_action = action_means+ tf.multiply(tf.random_normal(shape=[self.action_dim], mean=0, stddev=1), tf.exp(log_std))   # TODO 
    mvg=tf.contrib.distributions.MultivariateNormalDiag(action_means, tf.exp(log_std))
    self.logprob = mvg.log_prob(self.action_placeholder)

  def add_loss_op(self):
    self.loss = -tf.reduce_mean(self.logprob*self.advantage_placeholder)
  def add_optimizer_op(self):
    self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
  
  def add_baseline_op(self, scope = "baseline"):

    self.baseline = self.build_mlp(self.observation_placeholder, 1, scope,
                    self.config.n_layers, self.config.layer_size) # TODO
    self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=(None), name="baseline" )
    self.baselineLoss=tf.losses.mean_squared_error(labels=self.baseline_target_placeholder, predictions=self.baseline)
    self.update_baseline_op =tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.baselineLoss) # TODO

  def build(self):
    #build graph
    # add placeholders
    self.add_placeholders_op()
    # create policy net
    self.build_policy_network_op()
    # add square loss
    self.add_loss_op()
    # add optmizer for the main networks
    self.add_optimizer_op()
  
    if self.config.use_baseline:
      self.add_baseline_op()

    self.initialize()
  
  def initialize(self):
    # create tf session
    self.sess = tf.Session()
    # tensorboard stuff
    self.add_summary()
    # initiliaze all variables
    init = tf.global_variables_initializer()
    self.sess.run(init)
  
  def add_summary(self):
    
    # extra placeholders to log stuff from python
    self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
    self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")
  
    self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")
  
    # extra summaries from python -> placeholders
    tf.summary.scalar("Avg Reward 10kbatch", self.avg_reward_placeholder)
    tf.summary.scalar("Max Reward 10kbatch", self.max_reward_placeholder)
    tf.summary.scalar("Std Reward 10kbatch", self.std_reward_placeholder)
    tf.summary.scalar("Eval Reward 10kbatch", self.eval_reward_placeholder)
            
    # logging
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.output_path,self.sess.graph) 

  def init_averages(self):

    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.
  
  def update_averages(self, rewards, scores_eval):
    
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
  
    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]
  
  def record_summary(self, t):
  
    fd = {
      self.avg_reward_placeholder: self.avg_reward, 
      self.max_reward_placeholder: self.max_reward, 
      self.std_reward_placeholder: self.std_reward, 
      self.eval_reward_placeholder: self.eval_reward, 
    }
    summary = self.sess.run(self.merged, feed_dict=fd)
    # tensorboard stuff
    self.file_writer.add_summary(summary, t)
 
  def action(self, state, eps ):
    if rand.random()>eps:
      action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : state})[0]
    else:
      action=rand.random(2)*2-1
    sat=np.array([1,1])
    return np.clip(action, -sat, sat)
  
  def sample_path(self, env, num_episodes = None): 
    episode = 0
    episode_rewards = []
    paths = []
    t = 0
    while (num_episodes or t < self.config.batch_size):
      state = env.reset()
      states, actions, rewards = [], [], []
      episode_reward = 0
  
      for step in range(self.config.max_ep_len):
        states.append(state)
        action = action(self, state)
        print(action)
        state, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        t += 1
        if (done or step == self.config.max_ep_len-1):
          episode_rewards.append(episode_reward)  
          break
        if (not num_episodes) and t == self.config.batch_size:
          break
  
      path = {"observation" : np.array(states), 
                      "reward" : np.array(rewards), 
                      "action" : np.array(actions)}
      paths.append(path)
      episode += 1
      if num_episodes and episode >= num_episodes:
        break        
  
    return paths, episode_rewards
  
  def get_returns(self, paths):
    
    all_returns = []
    for path in paths:
      rewards = path["reward"]
      path_returns = [sum([r*config.gamma**i for i,r in enumerate(rewards[n:])]) for n in range(len(rewards))]  # TODO
      all_returns.append(path_returns)
    returns = np.concatenate(all_returns)
  
    return returns
  
  def calculate_advantage(self, returns, observations):
    
    adv = returns
    if self.config.use_baseline:
      adv=returns-self.sess.run(self.baseline, feed_dict={self.observation_placeholder: observations})
    
    if self.config.normalize_advantage:
      adv=(adv-np.mean(adv))/np.std(adv)
    return adv
  
  
  def update_baseline(self, returns, observations):
    self.sess.run(self.update_baseline_op, feed_dict={self.observation_placeholder: observations, 
                                                      self.baseline_target_placeholder: returns }) # TODO
  
  def train_network(self, paths):
    #last_eval = 0 
    #last_record = 0
    #scores_eval = []
    
    #self.init_averages()
    #scores_eval = [] # list of scores computed at iteration time
    print("TRAINING: PG")
    for t in range(1):
  
      # collect a minibatch of samples
      #paths, total_rewards = self.sample_path(self.env) 
      #scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      total_rewards=[sum(path["reward"]) for path in paths]
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)
      advantages = self.calculate_advantage(returns, observations)

      # run training operations
      if self.config.use_baseline:
        self.update_baseline(returns, observations)
      self.sess.run(self.train_op, feed_dict={
                    self.observation_placeholder : observations, 
                    self.action_placeholder : actions, 
                    self.advantage_placeholder : advantages})
  
      # tf stuff
      # if (t % self.config.summary_freq == 0):
      #   self.update_averages(total_rewards, scores_eval)
      #   self.record_summary(t)

      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      print("Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward))
      #self.logger.info(msg)
  
    #   if  self.config.record and (last_record > self.config.record_freq):
    #     self.logger.info("Recording...")
    #     last_record =0
    #     self.record()
  
    # self.logger.info("- Training done.")


if __name__ == "__main__":
    env = gym.make("sai2-v0")
    print(env.observation_space.shape)
