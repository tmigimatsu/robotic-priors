import ctypes as c

sai2py = c.cdll.LoadLibrary("./libsai2-env.so")

c_ubyte_p = c.POINTER(c.c_ubyte)
c_double_p = c.POINTER(c.c_double)

sai2_init = sai2py.init
sai2_init.restype = c.c_void_p
sai2_init.argtypes = [c.c_char_p, c.c_char_p, c.c_char_p, c.c_size_t, c.c_size_t]

sai2_step = sai2py.step
sai2_step.restype = c.c_bool
sai2_step.argtypes = [c.c_void_p, c_double_p, c_ubyte_p, c_double_p, c_double_p]

sai2_reset = sai2py.reset
sai2_reset.restype = None
sai2_reset.argtypes = [c.c_void_p, c_ubyte_p]

def init(world_file, robot_file, robot_name, window_width, window_height):
    return sai2_init(c.c_char_p(world_file.encode()),
                     c.c_char_p(robot_file.encode()),
                     c.c_char_p(robot_name.encode()),
                     c.c_size_t(window_width),
                     c.c_size_t(window_height))

def step(sai2_env, action, observation, info):
    reward = c.c_double()
    done = sai2_step(sai2_env,
                     action.ctypes.data_as(c_double_p),
                     observation.ctypes.data_as(c_ubyte_p),
                     c.byref(reward),
                     info.ctypes.data_as(c_double_p))
    return (reward.value, done)

def reset(sai2_env, observation):
    sai2_reset(sai2_env,
               observation.ctypes.data_as(c_ubyte_p))
