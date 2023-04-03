import numpy as np
import pandas as pd
import h5py

# Create HDF5 File
with h5py.File("./data.h5", "w") as hdf:

    hdf.create_group("Vivian")  # to create a group or "folder"
    hdf.create_group("Sam")
    hdf.create_group("John")
    hdf.create_group("dataset")

# Create csv (numpy arrays) datasets for each group
with h5py.File("./data.h5", "a") as hdf:

    vivian_group = hdf["Vivian"]
    sam_group = hdf["Sam"]
    john_group = hdf["John"]

    # John's Datasets Generation
    john_group.create_dataset(
        "back_right_J",
        data=np.genfromtxt("./John_Raw/Back Right Pocket Jump.csv", delimiter=","),
    )
    john_group.create_dataset(
        "back_right_W",
        data=np.genfromtxt("./John_Raw/Back Right Pocket Walk.csv", delimiter=","),
    )
    john_group.create_dataset(
        "front_coat_J",
        data=np.genfromtxt("./John_Raw/Front Coat Pocket Jump.csv", delimiter=","),
    )
    john_group.create_dataset(
        "front_coat_W",
        data=np.genfromtxt("./John_Raw/Front Coat Pocket Walk.csv", delimiter=","),
    )
    john_group.create_dataset(
        "left_coat_J",
        data=np.genfromtxt("./John_Raw/Left Coat Pocket Jump.csv", delimiter=","),
    )
    john_group.create_dataset(
        "left_coat_W",
        data=np.genfromtxt("./John_Raw/Left Coat Pocket Walk.csv", delimiter=","),
    )
    john_group.create_dataset(
        "left_hand_J",
        data=np.genfromtxt("./John_Raw/Left Hand Jump.csv", delimiter=","),
    )
    john_group.create_dataset(
        "left_hand_W",
        data=np.genfromtxt("./John_Raw/Left Hand Walk.csv", delimiter=","),
    )
    john_group.create_dataset(
        "left_pocket_J",
        data=np.genfromtxt("./John_Raw/Left Pocket Jump.csv", delimiter=","),
    )
    john_group.create_dataset(
        "left_pocket_W",
        data=np.genfromtxt("./John_Raw/Left Pocket Walk.csv", delimiter=","),
    )

    # Sam's Datasets Generation
    sam_group.create_dataset(
        "back_right_J",
        data=np.genfromtxt("./Sam_Raw/Back-right-jump.csv", delimiter=","),
    )
    sam_group.create_dataset(
        "back_right_W", data=np.genfromtxt("./Sam_Raw/Back-right.csv", delimiter=",")
    )
    sam_group.create_dataset(
        "front_coat_J",
        data=np.genfromtxt("./Sam_Raw/Front-coat-jump.csv", delimiter=","),
    )
    sam_group.create_dataset(
        "front_coat_W", data=np.genfromtxt("./Sam_Raw/Front-coat.csv", delimiter=",")
    )
    sam_group.create_dataset(
        "left_coat_J", data=np.genfromtxt("./Sam_Raw/Left-coat-jump.csv", delimiter=",")
    )
    sam_group.create_dataset(
        "left_coat_W", data=np.genfromtxt("./Sam_Raw/Left-coat.csv", delimiter=",")
    )
    sam_group.create_dataset(
        "left_hand_J", data=np.genfromtxt("./Sam_Raw/Left-hand-jump.csv", delimiter=",")
    )
    sam_group.create_dataset(
        "left_hand_W", data=np.genfromtxt("./Sam_Raw/Left-hand.csv", delimiter=",")
    )
    sam_group.create_dataset(
        "left_pocket_J",
        data=np.genfromtxt("./Sam_Raw/Left-pocket-jump.csv", delimiter=","),
    )
    sam_group.create_dataset(
        "left_pocket_W", data=np.genfromtxt("./Sam_Raw/Left-pocket.csv", delimiter=",")
    )

    # Vivian's Datasets Generation
    vivian_group.create_dataset(
        "back_right_J",
        data=np.genfromtxt("./Vivian_Raw/Right Back Jumping.csv", delimiter=","),
    )
    vivian_group.create_dataset(
        "back_right_W",
        data=np.genfromtxt("./Vivian_Raw/Right Back Walking.csv", delimiter=","),
    )
    vivian_group.create_dataset(
        "left_pocket_J",
        data=np.genfromtxt("./Vivian_Raw/Left Pocket Jumping.csv", delimiter=","),
    )
    vivian_group.create_dataset(
        "left_pocket_W",
        data=np.genfromtxt("./Vivian_Raw/Left Pants Walking.csv", delimiter=","),
    )
    vivian_group.create_dataset(
        "left_hand_J",
        data=np.genfromtxt("./Vivian_Raw/Left Hand Jumping.csv", delimiter=","),
    )
    vivian_group.create_dataset(
        "left_hand_W",
        data=np.genfromtxt("./Vivian_Raw/Left Hand Walking.csv", delimiter=","),
    )
    vivian_group.create_dataset(
        "left_coat_W",
        data=np.genfromtxt("./Vivian_Raw/Left Coat Walking.csv", delimiter=","),
    )
    vivian_group.create_dataset(
        "left_coat_J",
        data=np.genfromtxt("./Vivian_Raw/Left Coat Jumping.csv", delimiter=","),
    )
    vivian_group.create_dataset(
        "front_coat_W",
        data=np.genfromtxt("./Vivian_Raw/Coat Chest Walking.csv", delimiter=","),
    )
    vivian_group.create_dataset(
        "front_coat_J",
        data=np.genfromtxt("./Vivian_Raw/Coat Chest Jumping.csv", delimiter=","),
    )

    # Training the model:
    # Create 2 Pandas Dataframes: 1 with all of the Walking data, one with all of the Jumping data
    jump = pd.DataFrame()
    walk = pd.DataFrame()

    for title, data in vivian_group.items():
        if "W" in title:
            jump = pd.concat([jump, pd.DataFrame(data)])
        elif "J" in title:
            walk = pd.concat([walk, pd.DataFrame(data)])

    for title, data in sam_group.items():
        if "W" in title:
            jump = pd.concat([jump, pd.DataFrame(data)])
        elif "J" in title:
            walk = pd.concat([walk, pd.DataFrame(data)])

    for title, data in john_group.items():
        if "W" in title:
            jump = pd.concat([jump, pd.DataFrame(data)])
        elif "J" in title:
            walk = pd.concat([walk, pd.DataFrame(data)])

    print("********** JUMPING DF *************")
    print(jump)
    print("********** WALKING DF *************")
    print(walk)