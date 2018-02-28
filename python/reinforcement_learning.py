import numpy as np
import tensorflow as tf

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

