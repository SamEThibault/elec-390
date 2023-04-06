import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


######################## DATA STORING ########################

# Create HDF5 File
with h5py.File("./data.h5", "w") as hdf:

    hdf.create_group("Vivian")  # to create a group or "folder"
    hdf.create_group("Sam")
    hdf.create_group("John")
    hdf.create_group("dataset")

# Create 2 Pandas Dataframes: 1 with all of the Walking data, one with all of the Jumping data
# Add extra column for the labeling later
jump = pd.DataFrame(columns=np.arange(6))
walk = pd.DataFrame(columns=np.arange(6))

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

    # Filling in the 2 dataframes with all the jumping and walking data, respectively
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


######################## PRE-PROCESSING ########################

# Use isna() to check for missing values
# missing_values = walk.isna().sum()
# print(missing_values)

# missing_values = jump.isna().sum()
# print(missing_values)

# Labelling all columns
jump.columns = ["time", "x-g", "y-g", "z-g", "abs-g", "label"]
walk.columns = ["time", "x-g", "y-g", "z-g", "abs-g", "label"]

# Drop unnecessary columns
jump.drop(["time", "abs-g"], axis=1, inplace=True)
walk.drop(["time", "abs-g"], axis=1, inplace=True)

print("********** JUMPING DF (ORIGINAL) *************")
print(jump)
print("********** WALKING DF (ORIGNAL) *************")
print(walk)

# plt.plot(jump["time"], jump["y-g"])
# plt.xlabel("time (s)")
# plt.ylabel("y-acceleration (m/s^2)")
# plt.show()

# Replace missing values with the mean of the column
walk.fillna(walk.mean(), inplace=True)
jump.fillna(jump.mean(), inplace=True)

# Remove noise using a moving average filter                # Might not be a good window size, double check
jump = jump.rolling(25, min_periods=1).mean()
walk = walk.rolling(25, min_periods=1).mean()


print("********** JUMPING DF (FILTERED) *************")
print(jump)
print("********** WALKING DF (FILTERED) *************")
print(walk)

# plt.plot(jump["time"], jump["y-g"])
# plt.xlabel("time (s)")
# plt.ylabel("y-acceleration (m/s^2)")
# plt.show()

######################## FEATURE EXTRACTION ########################
# max, min, range, mean, median, variance, skewness, mode (most frequent number)

# Different Types of Features:
# Statistical: max, min, mean, standard deviation, median, range, skewness, kurtosis
# Frequency features: FFT, followed by amplitude or phase in certain regions
# Information: entropy, coherence
def calculate_features(data):
    features = {}
    features["max"] = np.max(data)
    features["min"] = np.min(data)
    features["range"] = features["max"] - features["min"]
    features["mean"] = np.mean(data)
    features["median"] = np.median(data)
    features["variance"] = np.var(data)
    features["skewness"] = skew(data)
    return features


# Extract features for each column in the both data frames
jump_column_features = {}
walk_column_features = {}

# import h5py
# import numpy as np
# from sklearn.utils import shuffle

# # Load the signals from the HDF5 file
# with h5py.File('your_file.hdf5', 'r') as f:
#     signals = f['signals'][:]
    
# # Divide each signal into 5-second windows
# window_size = 5 * 1000  # 5 seconds in milliseconds
# windows = []
# for signal in signals:
#     for i in range(0, len(signal) - window_size, window_size):
#         window = signal[i:i+window_size]
#         windows.append(window)
# windows = np.array(windows)

# # Shuffle the segmented data
# windows_shuffled = shuffle(windows, random_state=42)

# # Split the shuffled data into training and testing sets
# split_index = int(len(windows_shuffled) * 0.9)
# train_data = windows_shuffled[:split_index]
# test_data = windows_shuffled[split_index:]

# # Store the new dataset in the HDF5 file
# with h5py.File('your_file.hdf5', 'a') as f:
#     if 'segmented_data' in f:
#         del f['segmented_data']
#     segmented_data = f.create_dataset('segmented_data', data=windows_shuffled)
#     segmented_data.attrs['window_size'] = window_size
#     segmented_data.attrs['train_test_split'] = 0.9





# (500 data points per 5 seconds)
# jump_column_features : collection0, collection 1 : column0 : features, column1 : features...


# Find features for each collection of 500 data points
jump_split = np.array_split(jump, len(jump) / 500)
# print("JUMP SPLIT")
# print(jump_split)

# Extract features for each window of data in the jump dataframe
jump_features = []
jump_features_dict = {}
for i in range(0, len(jump), 500):
    window = jump[i:i+500]
    window_features = {}
    for column in window.columns:
        column_data = window[column]
        window_features[column] = calculate_features(column_data)
    jump_features.append(window_features)

# This allows us to associate the features dictionary with each collection index {index : features dict}
for i in range(len(jump_features)):
    jump_features_dict[i] = jump_features[i]

# Extract features for each window of data in the walk dataframe
walk_features = []
walk_features_dict = {}
for i in range(0, len(walk), 500):
    window = walk[i:i+500]
    window_features = {}
    for column in window.columns:
        column_data = window[column]
        window_features[column] = calculate_features(column_data)
    walk_features.append(window_features)

# This allows us to associate the features dictionary with each collection index {index : features dict}
for i in range(len(walk_features)):
    walk_features_dict[i] = walk_features[i]


# Normalize data so that it becomes suitable for logistic regression using StandardScaler
scaler = StandardScaler()
jump_scaled = scaler.fit_transform(jump)
walk_scaled = scaler.fit_transform(walk)

# Then make 5-second segments (should be splitting the array into sets of 500 data points), (DONE)
# Shuffle and split data into 90% training, 10% testing 
# Train logistic regression model to classify the data into walking and jumping classes


# Then apply that model on the test set, and record accuracy

# Deploy the final model in a desktop app
