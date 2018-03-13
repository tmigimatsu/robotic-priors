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
            # if D_actual >= D:
            #     break

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


def batch_data(data=None, size_batch=100, extra=False, filename=None, dataset="all", flatten=True):
    if data is not None:
        observations = data[1]
        aInd=data[6]
        sp_hat=data[7]
        d=data[8]


        try:
            mean_observation = np.load("../resources/mean_observation.npy")
        except:
            print("Could not load ../resources/mean_observation.npy")
            num_observations = sum(o.shape[0] for o in observations)
            mean_observation = sum(np.mean(o.astype(np.float64), axis=0) * (o.shape[0] / num_observations / 255) for o in observations)
            mean_observation = mean_observation[np.newaxis,...].astype(np.float32)

        for a, o, r, x, dx, s_hat, aindex, sp_hat, d in zip(*data): 
            # Preprocess observation
            o = o.astype(np.float32) / 255 - mean_observation
            if flatten:
                o = o.reshape(o.shape[0], -1)
                assert np.all(np.any(np.abs(o) > 0.01, axis=1))

            if extra:
                x = x[...,:2] - np.array([0, -0.45])[np.newaxis,:]
                dx = dx[...,:2]
                yield (o, a, r, x, dx, s_hat, aInd, sp_hat, d)
            else:
                yield (o, a, r)

        return

    if filename is None:
        filename = get_filename(-1)

    try:
        mean_observation = np.load("../resources/mean_observation.npy")
    except:
        pass
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

        o  = []
        a  = []
        r  = []
        if extra:
            x  = []
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
                o  = []
                a  = []
                r  = []
                if extra:
                    x  = []
                    dx = []
                T = 0


        # print("Dataset finished: " + dataset)

class DataLogger:

    def __init__(self):
        if not os.path.exists("results"):
            os.makedirs("results")
        self.filename = "results/data-{}.hdf5".format(strftime("%m-%d_%H-%M"), gmtime())
        self.f = h5py.File(self.filename, "w")

        self.actions_history = []
        self.observations_history = []
        self.rewards_history = []
        self.xs_history = []
        self.dxs_history = []
        self.learned_states_history = []

        self.aInds_states_history = []
        self.sp_hats_states_history = []
        self.donemasks_states_history = []

    def log_initial_observation(self, initial_observation):
        dset = self.f.create_dataset("initial_observation", initial_observation.shape, dtype=initial_observation.dtype)
        dset[...] = initial_observation

    def log(self, i, actions, observations, rewards, xs, dxs, learned_states, aInds, sp_hats, donemasks):
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
        dset = grp.create_dataset("learned_states", learned_states.shape, dtype=learned_states.dtype)
        dset[...] = learned_states

        dset = grp.create_dataset("aInds", aInds.shape, dtype=aInds.dtype)
        dset[...] = aInds
        dset = grp.create_dataset("sp_hats", sp_hats.shape, dtype=sp_hats.dtype)
        dset[...] = sp_hats
        dset = grp.create_dataset("donemasks", donemasks.shape, dtype=donemasks.dtype)
        dset[...] = donemasks




        # Add to history
        self.actions_history.append(actions)
        self.observations_history.append(observations)
        self.rewards_history.append(rewards)
        self.xs_history.append(xs)
        self.dxs_history.append(dxs)
        self.learned_states_history.append(learned_states)
        self.aInds_states_history.append(aInds)
        self.sp_hats_states_history.append(sp_hats)
        self.donemasks_states_history.append(donemasks)



    def flush(self):

        actions = self.actions_history
        observations = self.observations_history
        rewards = self.rewards_history
        xs = self.xs_history
        dxs = self.dxs_history
        learned_states = self.learned_states_history
        aInds= self.aInds_states_history
        sp_hats= self.sp_hats_states_history
        donemasks= self.donemasks_states_history



        self.actions_history = []
        self.observations_history = []
        self.rewards_history = []
        self.xs_history = []
        self.dxs_history = []
        self.learned_states_history = []
        self.aInds_states_history = []
        self.sp_hats_states_history=[]
        self.donemasks_states_history=[]

        self.f.flush()

        return actions, observations, rewards, xs, dxs, learned_states, aInds, sp_hats, donemasks

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.f.close()
