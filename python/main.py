#!/usr/bin/env python

import gym
import gym_sai2
import threading

import numpy as np
import matplotlib.pyplot as plt

def wait():
    input("Hit <enter> to close.\n")
    exit()

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == "__main__":
    thread = threading.Thread(target=wait, daemon=True)
    thread.start()

    # sai = SaiEnv("world.urdf", "kuka_iiwa.urdf", "kuka_iiwa")
    env = gym.make("sai2-v0")
    env.seed(0)
    agent = RandomAgent(env.action_space)

    num_episodes = 100
    reward = 0
    done = False

    for i in range(num_episodes):
        print("Iteration: {}".format(i))
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)

            if reward != 0:
                print("Reward: {}".format(reward))
            if done:
                break

    with open("log.txt", "w") as f:
        f.write("COMPLETE\n")

    env.close()

