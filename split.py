import numpy as np
import h5py

# data = [[back_right_J, ...]]
data = []

# 90% for training, 10% for testing
with h5py.File("./data.h5", "a") as hdf:

    dataset = hdf["dataset"]
    hdf.create_dataset()

    data.append(np.array_split(hdf.get('Vivian/back_right_J'), 6))