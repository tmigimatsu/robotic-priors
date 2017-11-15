import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym_sai2.envs import sai2_env_cpp
 
class SaiEnv(gym.Env):

    action_space = None
    observation_space = None

    def __init__(self):
        world_file = "../resources/world.urdf"
        robot_file = "../resources/kuka_iiwa.urdf"
        robot_name = "kuka_iiwa"
        window_width = 300
        window_height = 200
        self.img_buffer = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        self.sai2_env = sai2_env_cpp.init(world_file, robot_file, robot_name,
                                          window_width, window_height)

    def _step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode is
        reached, you are responsible for calling `reset()` to reset this
        environment's state.
        
        Accepts an action and returns a tuple (observation, reward, done, info).
        
        Args:
            action (object): an action provided by the environment
            
        Returns:
            observation (object): agent's observation of the current environment
            reward (float): amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        reward, done = sai2_env_cpp.step(self.sai2_env, action, self.img_buffer)
        return (self.img_buffer, reward, done, {})

    def _reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation of the space.
        """
        sai2_env_cpp.reset(self.sai2_env, self.img_buffer)

    def _render(self, mode="human", close=False):
        """
        Renders the environment. The set of supported modes varies per
        environment. (And some environments do not support rendering at all.) By
        convention, if mode is:

        - human: render to the current display or terminal and return nothing.
          Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3), representing
          RGB values for an x-by-y pixel image, suitable for turning into a
          video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines and
          ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes the
            list of supported modes. It's recommended to call super() in
            implementations to use the functionality of this method.
            
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                  ... # pop up a window and render
                else:
                  super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if mode == "rgb_array":
            return self.img_buffer
        return

    def _seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number
            generators. We want to capture all such seeds used in order to
            ensure that there aren't accidental correlations between multiple
            generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
            number generators. The first value in the list should be the "main"
            seed, or the value which a reproducer should pass to 'seed'. Often,
            the main seed equals the provided 'seed', but this won't be true if
            seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
