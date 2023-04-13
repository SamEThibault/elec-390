import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


######################## DATA STORING ########################
# helper function for train() to set up hdf5 file and dataframes
def setup_data():
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
            "back_right_W",
            data=np.genfromtxt("./Sam_Raw/Back-right.csv", delimiter=","),
        )
        sam_group.create_dataset(
            "front_coat_J",
            data=np.genfromtxt("./Sam_Raw/Front-coat-jump.csv", delimiter=","),
        )
        sam_group.create_dataset(
            "front_coat_W",
            data=np.genfromtxt("./Sam_Raw/Front-coat.csv", delimiter=","),
        )
        sam_group.create_dataset(
            "left_coat_J",
            data=np.genfromtxt("./Sam_Raw/Left-coat-jump.csv", delimiter=","),
        )
        sam_group.create_dataset(
            "left_coat_W", data=np.genfromtxt("./Sam_Raw/Left-coat.csv", delimiter=",")
        )
        sam_group.create_dataset(
            "left_hand_J",
            data=np.genfromtxt("./Sam_Raw/Left-hand-jump.csv", delimiter=","),
        )
        sam_group.create_dataset(
            "left_hand_W", data=np.genfromtxt("./Sam_Raw/Left-hand.csv", delimiter=",")
        )
        sam_group.create_dataset(
            "left_pocket_J",
            data=np.genfromtxt("./Sam_Raw/Left-pocket-jump.csv", delimiter=","),
        )
        sam_group.create_dataset(
            "left_pocket_W",
            data=np.genfromtxt("./Sam_Raw/Left-pocket.csv", delimiter=","),
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

        return walk, jump


######################## FEATURE EXTRACTION ########################
def calculate_features(data):

    # will return dataframe with each acc dimension's features
    features = pd.DataFrame(
        columns=[
            "max_X",
            "max_Y",
            "max_Z",
            "min_X",
            "min_Y",
            "min_Z",
            "range_X",
            "range_Y",
            "range_Z",
            "mean_X",
            "mean_Y",
            "mean_Z",
            "median_X",
            "median_Y",
            "median_Z",
            "var_X",
            "var_Y",
            "var_Z",
            "skew_X",
            "skew_Y",
            "skew_Z",
            "kurtosis_X",
            "kurtosis_Y",
            "kurtosis_Z",
            "deviation_X",
            "deviation_Y",
            "deviation_Z",
            "quantile_X",
            "quantile_Y",
            "quantile_Z",
        ]
    )

    # 500 samples: 5 second intervals @ 100 Hz sampling rate, which is what Phyphox uses
    # split into 500 sample groups and find the features
    maximum = data.rolling(window=500).max()
    minimum = data.rolling(window=500).min()
    rng = maximum - minimum
    mean = data.rolling(window=500).mean()
    median = data.rolling(window=500).median()
    variance = data.rolling(window=500).var()
    skewness = data.rolling(window=500).skew()
    kurtosis = data.rolling(window=500).kurt()
    deviation = data.rolling(window=500).std()
    quantile = data.rolling(window=500).quantile(0.5)

    # then, split into respective dimensions
    features["max_X"], features["max_Y"], features["max_Z"] = (
        maximum.iloc[:, 0],
        maximum.iloc[:, 1],
        maximum.iloc[:, 2],
    )
    features["min_X"], features["min_Y"], features["min_Z"] = (
        minimum.iloc[:, 0],
        minimum.iloc[:, 1],
        minimum.iloc[:, 2],
    )
    features["range_X"], features["range_Y"], features["range_Z"] = (
        rng.iloc[:, 0],
        rng.iloc[:, 1],
        rng.iloc[:, 2],
    )
    features["mean_X"], features["mean_Y"], features["mean_Z"] = (
        mean.iloc[:, 0],
        mean.iloc[:, 1],
        mean.iloc[:, 2],
    )
    features["median_X"], features["median_Y"], features["median_Z"] = (
        median.iloc[:, 0],
        median.iloc[:, 1],
        median.iloc[:, 2],
    )
    features["var_X"], features["var_Y"], features["var_Z"] = (
        variance.iloc[:, 0],
        variance.iloc[:, 1],
        variance.iloc[:, 2],
    )
    features["skew_X"], features["skew_Y"], features["skew_Z"] = (
        skewness.iloc[:, 0],
        skewness.iloc[:, 1],
        skewness.iloc[:, 2],
    )
    features["kurtosis_X"], features["kurtosis_Y"], features["kurtosis_Z"] = (
        kurtosis.iloc[:, 0],
        kurtosis.iloc[:, 1],
        kurtosis.iloc[:, 2],
    )
    features["deviation_X"], features["deviation_Y"], features["deviation_Z"] = (
        deviation.iloc[:, 0],
        deviation.iloc[:, 1],
        deviation.iloc[:, 2],
    )
    features["quantile_X"], features["quantile_Y"], features["quantile_Z"] = (
        quantile.iloc[:, 0],
        quantile.iloc[:, 1],
        quantile.iloc[:, 2],
    )

    # drop none values (size will dictate size of result)
    features.dropna(inplace=True)
    return features


# used in app.py
def train():

    walk, jump = setup_data()

    ######################## PRE-PROCESSING ########################
    # name labels
    jump.columns = [
        "time",
        "x-g",
        "y-g",
        "z-g",
        "abs-g",
        "label",
    ]
    walk.columns = [
        "time",
        "x-g",
        "y-g",
        "z-g",
        "abs-g",
        "label",
    ]

    # Drop unnecessary columns
    jump.drop(["time", "abs-g", "label"], axis=1, inplace=True)
    walk.drop(["time", "abs-g", "label"], axis=1, inplace=True)

    # drop noneType values
    walk.dropna(inplace=True)
    jump.dropna(inplace=True)

    # Remove noise using a moving average filter, window size: 40
    jump = jump.rolling(40, min_periods=1).mean()
    walk = walk.rolling(40, min_periods=1).mean()

    ############## TRAINING #############

    # Train logistic regression model to classify the data into walking and jumping classes - John & Vivian

    # merge 2 datasets together, then put zeros as labels for walking, and ones for jump labels
    data_temp = [walk, jump]
    data = pd.concat(data_temp)
    labels = np.concatenate((np.zeros(len(walk)), np.ones(len(jump))))

    # get the features
    features = calculate_features(data)

    # split and shuffle the features into 90% training, 10% testing (use length of features that have dropped noneType vals)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels[: len(features)], test_size=0.1, shuffle=True, random_state=0
    )

    # initiate logistic regression model and pipeline it with scaled data
    l_reg = LogisticRegression(max_iter=10000)
    clf = make_pipeline(StandardScaler(), l_reg)
    clf.fit(X_train, y_train)

    # predictions and probabilities
    y_pred = clf.predict(X_test)
    y_clf_prob = clf.predict_proba(X_test)

    print("y_pred is: ", y_pred)
    print("y_clf_prob is: ", y_clf_prob)

    acc = accuracy_score(y_test, y_pred)
    print("accuracy is: ", acc)

    # Return regression object for application classifier
    return clf


# get data from csv input, and use trained model from when the app started up
def classify(data_original, model):

    data = data_original.copy()

    # label columns
    data.columns = ["time", "x-g", "y-g", "z-g", "abs-g"]

    # pre-processing
    # Drop unnecessary columns
    data.drop(["time", "abs-g"], axis=1, inplace=True)

    # normalize data and get features: then predict using passed in model
    data = data.rolling(25, min_periods=1).mean()
    features = calculate_features(data)
    res = model.predict(features)

    # ignore number of NA vals that were dropped in feature extraction
    data = data[: len(features)]

    data["label"] = ""

    # 0 means walking, 1 means jumping: replace these values with the proper labels
    data["label"] = ["walking" if x == 0 else "jumping" for x in res]

    return data
