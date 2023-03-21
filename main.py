import numpy as np
import h5py

# Create HDF5 File
with h5py.File("./data.h5", "w") as hdf:

    hdf.create_group("Vivian") # to create a group or "folder"
    hdf.create_group("Sam")
    hdf.create_group("John")
    hdf.create_group("dataset")

vivian_group, sam_group, john_group = None

with h5py.File("./data.h5", "r") as hdf:
    vivian_group = hdf["Vivian"]
    sam_group = hdf["Sam"]
    john_group = hdf["John"]


with h5py.File("./data.h5", "a") as hdf:

    # John's Datasets Generation 
    hdf.create_dataset("back_right_J", data=np.genfromtxt('./John_Raw/Back Right Pocket Jump.csv', delimiter=','))
