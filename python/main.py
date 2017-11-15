#!/usr/bin/env python

import gym
import gym_sai2
import threading

import numpy as np

def loop():
    while True:
        line = input()
        if line == "x":
            return

if __name__ == "__main__":
    # sai = SaiEnv("world.urdf", "kuka_iiwa.urdf", "kuka_iiwa")
    env = gym.make("sai2-v0")

    action = np.zeros(3)
    observation, reward, done, info = env.step(action)
    print(observation)
    print(reward)
    print(done)

    thread = threading.Thread(target=loop)
    thread.start()
