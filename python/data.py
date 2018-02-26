"""
Utility functions for processing data.
"""
import os
import h5py
import numpy as np
import math
import glob
from time import strftime, gmtime

def list_files():
    filepath = os.path.join("data")
    files = glob.glob(os.path.join(filepath, "*.hdf5"))
    files = sorted(files, key=lambda f: os.stat(f).st_mtime)
    return files

def get_filename(idx):
    filename = list_files()[idx]
    print("Loading: ", filename)
    return filename

def get_mean_observation(filename):
    f = h5py.File(filename, "r")

    # Calculate mean observation
    if not "mean_observation" in f:
        dim_o = np.array(f["episodes/00000/observations"].shape)[1:]
        
        # Initialize mean
        D = 100
        D_actual = 0
        mean_observation = np.zeros(dim_o)

        # Iterate over episodes
        episodes = f["episodes"]
        for i in episodes:
            try:
                observations = episodes[i]["observations"][()]
            except:
                # Corrupt data
                continue

            observation = (observations.astype(np.float64) / 255).mean(axis=0)
            mean_observation += observation / D
            D_actual += 1
            if D_actual >= D:
                break

        mean_observation *= D / D_actual

        # Write mean observation to file
        f.close()
        with h5py.File(filename, "a") as f:
            dset = f.create_dataset("mean_observation", mean_observation.shape, dtype=mean_observation.dtype)
            dset[...] = mean_observation

    else:
        mean_observation = f["mean_observation"][()]
        f.close()
    return mean_observation

def get_observation_dim(filename=None):
    if filename is None:
        filename = get_filename(-1)

    mean_observation = get_mean_observation(filename)
    return mean_observation.shape

def raw_data(size_batch=100, filename=None):
    if filename is None:
        filename = get_filename(-1)

    f = h5py.File(filename, "r")

    episodes = f["episodes"]

    o = []
    T = 0
    for i in range(len(episodes)):
        grp = episodes["{0:05d}".format(i)]
        observations = grp["observations"][()].astype(np.float32)

        # Append data to lists
        o.append(observations)
        T += observations.shape[0]

        if T >= size_batch:
            # Gather batch
            o = np.concatenate(o, axis=0)
            return o[:size_batch]


def batch_data(size_batch=100, extra=False, filename=None, dataset="all", flatten=True):
    if filename is None:
        filename = get_filename(-1)

    mean_observation = get_mean_observation(filename)
    f = h5py.File(filename, "r")

    episodes = f["episodes"]
    if dataset == "train":
        D = (0, math.floor(0.8 * len(episodes)))
        size_batch = min(1000 * D[1], size_batch)
    elif dataset == "test":
        D = (math.floor(0.8 * len(episodes)), len(episodes))
        size_batch = min(1000 * (D[1] - D[0]), size_batch)
    else:
        D = (0, len(episodes))
        size_batch = min(1000 * D[1], size_batch)
    print(dataset, D, len(episodes))

    while True:

        if D[0] == 0:
            o_0 = f["initial_observation"][()].astype(np.float32)
            if extra:
                x_0 = np.zeros(2, dtype=np.float32)
        else:
            try:
                o_0 = episodes["{0:05d}/observations".format(D[0]-1)][-1,...].astype(np.float32)
                if extra:
                    x_0 = episodes["{0:05d}/xs".format(D[0]-1)][-1,...].astype(np.float32)
                    x_0 -= np.array([0, -0.45], dtype=np.float32)[np.newaxis,:]
            except:
                o_0 = f["initial_observation"][()].astype(np.float32)
                if extra:
                    x_0 = np.zeros(2, dtype=np.float32)
        o_0 = o_0 / 255 - mean_observation
        if flatten:
            o_0 = o_0.reshape(-1)
        o_0 = o_0[np.newaxis,...]
        o  = [o_0]
        a  = []
        r  = []
        if extra:
            x_0 = x_0[np.newaxis,:]
            x  = [x_0]
            dx = []
        T = 0

        for i in range(D[0], D[1]):
            try:
                grp = episodes["{0:05d}".format(i)]
                actions      = grp["actions"][()].astype(np.float32)
                observations = grp["observations"][()].astype(np.float32)
                rewards      = grp["rewards"][()].astype(np.float32)
                if extra:
                    xs  = grp["xs"][()].astype(np.float32)
                    dxs = grp["dxs"][()].astype(np.float32)
            except:
                continue

            # Flatten observations and center
            observations = observations / 255 - mean_observation[np.newaxis,...]
            if flatten:
                observations = observations.reshape(observations.shape[0],-1)
                assert np.all(np.any(np.abs(observations) > 0.01, axis=1))

            # Append data to lists
            o.append(observations)
            a.append(actions)
            r.append(rewards)
            if extra:
                x.append(xs[...,:2] - np.array([0, -0.45], dtype=np.float32)[np.newaxis,:])
                dx.append(dxs[...,:2])
            T += observations.shape[0]

            if T >= size_batch:
                # Gather batch
                o  = np.concatenate(o, axis=0)
                a  = np.concatenate(a, axis=0)
                r  = np.concatenate(r, axis=0)
                if extra:
                    x  = np.concatenate(x, axis=0)
                    dx = np.concatenate(dx, axis=0)

                # Yield batch
                if dataset == "test":
                    while True:
                        if extra:
                            yield (o, a, r, x, dx)
                        else:
                            yield (o, a, r)
                else:
                    if extra:
                        yield (o, a, r, x, dx)
                    else:
                        yield (o, a, r)
                o_0 = o[-1,np.newaxis,...]
                o  = [o_0]
                a  = []
                r  = []
                if extra:
                    x_0 = x[-1,np.newaxis,:]
                    x  = [x_0]
                    dx = []
                T = 0


        # print("Dataset finished: " + dataset)

class DataLogger:

    def __init__(self):
        if not os.path.exists("results"):
            os.makedirs("results")
        self.filename = "results/data-{}.hdf5".format(strftime("%m-%d_%H-%M"), gmtime())
        self.f = h5py.File(self.filename, "w")

    def log_initial_observation(self, initial_observation):
        dset = self.f.create_dataset("initial_observation", initial_observation.shape, dtype=initial_observation.dtype)
        dset[...] = initial_observation

    def log(self, i, actions, observations, rewards, xs, dxs):
        # Vectorize trajectory
        actions = np.row_stack(actions)
        observations = np.concatenate(observations, axis=0)
        rewards = np.array(rewards)
        xs = np.row_stack(xs)
        dxs = np.row_stack(dxs)

        # Save trajectory to dataset
        grp = self.f.create_group("/episodes/{0:05d}".format(i))
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


    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.f.close()
