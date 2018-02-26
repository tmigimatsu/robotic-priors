#!/usr/bin/env python

import gym
import gym_sai2

import numpy as np
from data import *
from robotic_priors import *
from reinforcement_learning import *
import threading

def wait():
    input("Hit <enter> to close.\n")
    exit()

if __name__ == "__main__":
    SIZE_BATCH = 10
    LEN_TRAJECTORY  = 1000
    NUM_ITERATIONS = 20
    NUM_RP_ITERATIONS = 10

    # Create thread to kill process (ctrl-c doesn't work?)
    thread = threading.Thread(target=wait, daemon=True)
    thread.start()

    # Create sai2 environment
    env = gym.make("sai2-v0")
    env.seed(0)
    initial_observation = env.reset()

    # State representation learning
    robotic_priors = RoboticPriors(initial_observation.shape)
    robotic_priors.reset_session()

    # Reinforcement learning agent
    agent = RandomAgent(env.action_space)

    for i in range(NUM_ITERATIONS):

        # Generate RL trajectories
        with DataLogger() as d:
            filename = d.filename

            # Get initial observation
            ob, reward, done = np.array(initial_observation)[np.newaxis,...], 0, False
            d.log_initial_observation(initial_observation)

            for i in range(SIZE_BATCH):

                print("Iteration: {}".format(i))

                actions, observations, rewards, xs, dxs = [], [], [], [], []

                # Generate trajectory
                for _ in range(LEN_TRAJECTORY):
                    # Query RL policy
                    s_hat = robotic_priors.evaluate(np.reshape(ob, (1,-1)))
                    action = agent.act(s_hat, reward, done)
                    ob, reward, done, info = env.step(action)
                    ob = np.array(ob)[np.newaxis,...]

                    # Append to trajectory
                    actions.append(np.array(action))
                    observations.append(ob)
                    rewards.append(reward)
                    xs.append(np.array(info["x"]))
                    dxs.append(np.array(info["dx"]))

                d.log(i, actions, observations, rewards, xs, dxs)

        # Train represnetation learning
        robotic_priors.create_logger()
        train_batch = batch_data(size_batch=1000, extra=True, filename=filename, flatten=True)
        robotic_priors.train_network(NUM_RP_ITERATIONS, train_batch)

    env.close()

