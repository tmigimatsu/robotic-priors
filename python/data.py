import os
import h5py
import numpy as np
import math
import glob

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

def batch_data(size_batch=100, extra=False, filename=None, dataset="train"):
    if filename is None:
        filename = get_filename(-1)

    mean_observation = get_mean_observation(filename)
    f = h5py.File(filename, "r")

    episodes = f["episodes"]
    if dataset == "train":
        D = (0, math.floor(0.8 * len(episodes)))
        size_batch = min(D[1], size_batch)
    elif dataset == "test":
        D = (math.floor(0.8 * len(episodes)), len(episodes))
        size_batch = min(D[1] - D[0], size_batch)
    else:
        raise ValueError("Invalid dataset type: " + dataset)
    print(dataset, D, len(episodes))

    while True:

        if D[0] == 0:
            o_0 = f["initial_observation"][()].astype(np.float32)
        else:
            try:
                o_0 = episodes["{0:05d}/observations".format(D[0]-1)][-1,...].astype(np.float32)
            except:
                o_0 = f["initial_observation"][()].astype(np.float32)
        o_0 = o_0.reshape(1,-1) / 255
        o_0 -= mean_observation.reshape(1,-1)
        o  = [o_0]
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
            observations = observations.reshape(observations.shape[0],-1) / 255
            observations -= mean_observation.reshape(1,-1)
            assert np.all(np.any(np.abs(observations) > 0.01, axis=1))

            # Append data to lists
            o.append(observations)
            a.append(actions)
            r.append(rewards)
            if extra:
                x.append(xs[...,:2] - np.array([0, -0.45])[np.newaxis,:])
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
                while True:
                    if extra:
                        yield (o, a, r, x, dx)
                    else:
                        yield (o, a, r)

        print("Dataset finished: " + dataset)
