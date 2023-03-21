import numpy as np
import h5py

with h5py.File("./data.h5", "w") as hdf:

    hdf.create_group("Vivian") # to create a group or "folder"
    hdf.create_group("Sam")
    hdf.create_group("John")
    hdf.create_group("dataset")

    items = list(hdf.items())
    print(items)

    # hdf.create_dataset(
    #     "dataset1", data=matrix
    # )  # where matrix is a np.random.random() object
