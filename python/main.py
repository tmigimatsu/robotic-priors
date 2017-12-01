#!/usr/bin/env python

import gym
import gym_sai2
import threading

import numpy as np
import matplotlib.pyplot as plt
import h5py
from time import gmtime, strftime

def wait():
    input("Hit <enter> to close.\n")
    exit()

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == "__main__":
    NUM_EPISODES = 10

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()

    env = gym.make("sai2-v0")
    env.seed(0)
    agent = RandomAgent(env.action_space)

    filename = "data/data-{}.hdf5".format(strftime("%m-%d_%H-%M"), gmtime())
    with h5py.File(filename, "w") as f:

        for i in range(NUM_EPISODES):

            print("Iteration: {}".format(i))
            grp = f.create_group("/episodes/{0:05d}".format(i))

            ob = env.reset()
            reward = 0
            done = False

            actions = []
            observations = [np.array(ob)[np.newaxis,...]]
            rewards = []
            xs = []
            dxs = []

            while not done:
                action = agent.act(ob, reward, done)
                ob, reward, done, info = env.step(action)

                # if reward != 0:
                #     print("Reward: {}".format(reward))

                actions.append(np.array(action))
                observations.append(np.array(ob)[np.newaxis,...])
                rewards.append(reward)
                xs.append(np.array(info["x"]))
                dxs.append(np.array(info["dx"]))

            actions = np.row_stack(actions)
            observations = np.concatenate(observations, axis=0)
            rewards = np.array(rewards)
            xs = np.row_stack(xs)
            dxs = np.row_stack(dxs)

            dset = grp.create_dataset("actions", actions.shape, dtype=actions.dtype)
            dset[...] = actions
            dset = grp.create_dataset("observations", observations.shape, dtype=observations.dtype)
            dset[...] = observations
            dset = grp.create_dataset("rewards", rewards.shape, dtype=rewards.dtype)
            dset[...] = rewards
            dset = grp.create_dataset("xs", xs.shape, dtype=xs.dtype)
            dset[...] = xs
            dset = grp.create_dataset("dxs", dxs.shape, dtype=dxs.dtype)
            dset[...] = dxs

        env.close()

