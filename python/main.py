#!/usr/bin/env python

import gym
import gym_sai2

import numpy as np
from data import *
from robotic_priors import *
from reinforcement_learning import DQN_Agent
from configs.dqn import config
import threading

def wait():
    input("Hit <enter> to close.\n")
    exit()

if __name__ == "__main__":
    NUM_EPISODES = 10
    LEN_EPISODE  = 1000
    NUM_ITERATIONS = 20

    # Create thread to kill process (ctrl-c doesn't work?)
    thread = threading.Thread(target=wait, daemon=True)
    thread.start()

    # Create sai2 environment
    env = gym.make("sai2-v0")
    # TODO: Implement environment seed
    env.seed(0)

    # State representation learning
    robotic_priors = RoboticPriors(env.dim_observation)
    robotic_priors.reset_session()

    # Reinforcement learning agent
    agent = DQN_Agent(env.action_space)

    for i in range(NUM_ITERATIONS):

        print("Iteration: {}".format(i))

        # Generate RL trajectories
        with DataLogger() as d:
            filename = d.filename

            for j in range(NUM_EPISODES):

                print("Episode: {}".format(j))

                # Get initial observation
                ob = env.reset()
                ob = np.array(ob)[np.newaxis,...]

                # One episode
                actions, observations, rewards, xs, dxs, learned_states = [], [ob], [], [], [], []

                # Generate trajectory
                for _ in range(LEN_EPISODE):
                    # Query RL policy
                    s_hat = robotic_priors.evaluate(np.reshape(ob, (1,-1)))
                    action, actionInd = agent.action(s_hat)
                    ob, reward, done, info = env.step(action)
                    ob = np.array(ob)[np.newaxis,...]

                    # Append to trajectory
                    actions.append(np.array(actionInd)) # save index because one hot stuff
                    observations.append(ob)
                    rewards.append(reward)
                    xs.append(np.array(info["x"]))
                    dxs.append(np.array(info["dx"]))
                    learned_states.append(s_hat)

                # Vectorize trajectory
                actions = np.row_stack(actions)
                observations = np.concatenate(observations, axis=0)
                rewards = np.array(rewards)
                xs = np.row_stack(xs)
                dxs = np.row_stack(dxs)
                learned_states = np.row_stack(learned_states)

                # Log data
                d.log(j, actions, observations, rewards, xs, dxs, learned_states)

            # Get list of actions, observations, rewards, xs, dxs, learned_states from all episodes
            episodes = d.flush()

        # Train represnetation learning
        robotic_priors.create_logger()
        robotic_priors_data_generator = batch_data(data=episodes, extra=True, flatten=True)
        robotic_priors.train_network(robotic_priors_data_generator)

    env.close()

