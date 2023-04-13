import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, mode
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
# max, min, range, mean, median, variance, skewness, mode (most frequent number)

# Different Types of Features:
# Statistical: max, min, mean, standard deviation, median, range, skewness, kurtosis
# Frequency features: FFT, followed by amplitude or phase in certain regions
# Information: entropy, coherence


def SplitFeature(df):
    return df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]


def calculate_features(data):
    # features = {}
    # features["max"] = np.max(data)
    # features["min"] = np.min(data)
    # features["range"] = features["max"] - features["min"]
    # features["mean"] = np.mean(data)
    # features["median"] = np.median(data)
    # features["variance"] = np.var(data)
    # features["skewness"] = skew(data)
    # features["kurtosis"] = kurtosis(data)
    # features["deviation"] = np.std(data)
    # features["mode"] = mode(data)
    # return features

    features = pd.DataFrame(
        columns=[
            "meanX",
            "meanY",
            "meanZ",
            "stdX",
            "stdY",
            "stdZ",
            "maxX",
            "maxY",
            "maxZ",
            "minX",
            "minY",
            "minZ",
            "kurtX",
            "kurtY",
            "kurtZ",
            "skewX",
            "skewY",
            "skewZ",
            "rmsX",
            "rmsY",
            "rmsZ",
            "medianX",
            "medianY",
            "medianZ",
            "diffX",
            "diffY",
            "diffZ",
            "diffabsX",
            "diffabsY",
            "diffabsZ",
            "varianceX",
            "varianceY",
            "varianceZ",
        ]
    )

    window_size = 500

    def difference(x):
        return x.iloc[-1] - x.iloc[0]

    def difference_abs(x):
        return abs(x.iloc[-1] - x.iloc[0])

    mean = data.rolling(window=window_size).mean()
    std = data.rolling(window=window_size).std()
    max = data.rolling(window=window_size).max()
    min = data.rolling(window=window_size).min()
    kurt = data.rolling(window=window_size).kurt()
    skew = data.rolling(window=window_size).skew()
    rms = data.pow(2).rolling(window=window_size).mean().apply(np.sqrt, raw=True)
    median = data.rolling(window=window_size).median()
    diff = data.rolling(window=window_size).apply(difference)
    diff_abs = data.rolling(window=window_size).apply(difference_abs)
    variance = data.rolling(window=window_size).std() ** 2

    features["meanX"], features["meanY"], features["meanZ"] = SplitFeature(mean)
    features["stdX"], features["stdY"], features["stdZ"] = SplitFeature(std)
    features["maxX"], features["maxY"], features["maxZ"] = SplitFeature(max)
    features["minX"], features["minY"], features["minZ"] = SplitFeature(min)
    features["kurtX"], features["kurtY"], features["kurtZ"] = SplitFeature(kurt)
    features["skewX"], features["skewY"], features["skewZ"] = SplitFeature(skew)
    features["rmsX"], features["rmsY"], features["rmsZ"] = SplitFeature(rms)
    features["medianX"], features["medianY"], features["medianZ"] = SplitFeature(median)
    features["diffX"], features["diffY"], features["diffZ"] = SplitFeature(diff)
    features["diffabsX"], features["diffabsY"], features["diffabsZ"] = SplitFeature(
        diff_abs
    )
    features["varianceX"], features["varianceY"], features["varianceZ"] = SplitFeature(
        variance
    )

    features = features.dropna()
    return features


# used in app.py
def train():

    walk, jump = setup_data()

    ######################## PRE-PROCESSING ########################
    # Labelling all columns
    # jump.columns = ["time", "x-g", "y-g", "z-g", "abs-g", "label"]
    # walk.columns = ["time", "x-g", "y-g", "z-g", "abs-g", "label"]

    # # Drop unnecessary columns
    # jump.drop(["time", "abs-g"], axis=1, inplace=True)
    # walk.drop(["time", "abs-g"], axis=1, inplace=True)

    jump.columns = [
        "time",
        "x-g",
        "y-g",
        "z-g",
        "abs-g",
        "label",
    ]  # john edit - removed label
    walk.columns = [
        "time",
        "x-g",
        "y-g",
        "z-g",
        "abs-g",
        "label",
    ]  # talk to sam ab this

    # Drop unnecessary columns
    jump.drop(["time", "abs-g", "label"], axis=1, inplace=True)
    walk.drop(["time", "abs-g", "label"], axis=1, inplace=True)

    # print("********** JUMPING DF (ORIGINAL) *************")
    # print(jump)
    # print("********** WALKING DF (ORIGNAL) *************")
    # print(walk)

    # Replace missing values with the mean of the column
    walk.fillna(walk.mean(), inplace=True)
    jump.fillna(jump.mean(), inplace=True)

    # Remove noise using a moving average filter                # Might not be a good window size, double check
    jump = jump.rolling(25, min_periods=1).mean()
    walk = walk.rolling(25, min_periods=1).mean()

    # print("********** JUMPING DF (FILTERED) *************")
    # print(jump)
    # print("********** WALKING DF (FILTERED) *************")
    # print(walk)

    # Shuffle the data
    jump_shuffled = jump.sample(frac=1).reset_index(drop=True)
    walk_shuffled = walk.sample(frac=1).reset_index(drop=True)

    # Shuffle and split data into 90% training, 10% testing
    jump_train = jump_shuffled[: int(0.9 * len(jump_shuffled))]
    jump_test = jump_shuffled[int(0.1 * len(jump_shuffled)) :]

    walk_train = walk_shuffled[: int(0.9 * len(walk_shuffled))]
    walk_test = walk_shuffled[int(0.1 * len(walk_shuffled)) :]

    # Extract features for each window of data in the jump dataframe
    # jump_features = []
    # jump_features_dict = {}
    # for i in range(0, len(jump_shuffled), 500):
    #     window = jump_shuffled[i : i + 500]
    #     window_features = {}
    #     for column in window.columns:
    #         column_data = window[column]
    #         window_features[column] = calculate_features(column_data)
    #     jump_features.append(window_features)

    # # This allows us to associate the features dictionary with each collection index {index : features dict}
    # for i in range(len(jump_features)):
    #     jump_features_dict[i] = jump_features[i]

    # # Extract features for each window of data in the walk dataframe
    # walk_features = []
    # walk_features_dict = {}
    # for i in range(0, len(walk_shuffled), 500):
    #     window = walk_shuffled[i : i + 500]
    #     window_features = {}
    #     for column in window.columns:
    #         column_data = window[column]
    #         window_features[column] = calculate_features(column_data)
    #     walk_features.append(window_features)

    # # This allows us to associate the features dictionary with each collection index {index : features dict}
    # for i in range(len(walk_features)):
    #     walk_features_dict[i] = walk_features[i]

    # Normalize shuffled training data so that it becomes suitable for logistic regression using StandardScaler
    # scaler = StandardScaler()
    # jump_train_scaled = scaler.fit_transform(jump_train)
    # walk_train_scaled = scaler.fit_transform(walk_train)

    ############## TRAINING #############
    # Train logistic regression model to classify the data into walking and jumping classes - John & Vivian

    data_temp = [walk, jump]
    data = pd.concat(data_temp)

    features = calculate_features(data)
    labels = np.concatenate((np.zeros(len(walk)), np.ones(len(jump))))

    print(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels[-len(features) :], test_size=0.1, shuffle=True, random_state=0
    )
    l_reg = LogisticRegression(max_iter=10000)
    clf = make_pipeline(StandardScaler(), l_reg)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_clf_prob = clf.predict_proba(X_test)
    X_pred = clf.predict(X_train)
    X_pred_prob = clf.predict_proba(X_train)

    print("y_pred is: ", y_pred)
    print("y_clf_prob is: ", y_clf_prob)

    acc = accuracy_score(y_test, y_pred)
    print("accuracy is: ", acc)

    print("X_pred is: ", X_pred)
    print("X_pred_prob is: ", X_pred_prob)

    acc_X = accuracy_score(y_train, X_pred)
    print("accuracy is: ", acc_X)

    return clf

    # Concatenate the jump and walk training data and labels
    # X_train = np.concatenate((jump_train_scaled, walk_train_scaled))
    # y_train = np.concatenate(
    #     (np.ones(len(jump_train_scaled)), np.zeros(len(walk_train_scaled)))
    # )

    # # Initialize logistic regression model
    # logreg_model = LogisticRegression()

    # # Fit the model to the training data
    # logreg_model.fit(X_train, y_train)

    # # Normalize the test data
    # jump_test_scaled = scaler.transform(jump_test)
    # walk_test_scaled = scaler.transform(walk_test)

    # # Concatenate the jump and walk test data and labels
    # X_test = np.concatenate((jump_test_scaled, walk_test_scaled))
    # y_test = np.concatenate(
    #     (np.ones(len(jump_test_scaled)), np.zeros(len(walk_test_scaled)))
    # )

    # # Make predictions on the test data using the trained model
    # y_pred = logreg_model.predict(X_test)

    # # Calculate accuracy of the predictions
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)
    return logreg_model


# get data from csv input, and use trained model from when the app started up
def classify(data_original, model):

    # add empty label column
    data = data_original.copy()

    # label columns
    data.columns = ["time", "x-g", "y-g", "z-g", "abs-g"]

    # pre-processing
    # Drop unnecessary columns
    data.drop(["time", "abs-g"], axis=1, inplace=True)

    # data.fillna(data.mean(), inplace=True)
    data = data.rolling(25, min_periods=1).mean()

    # data.dropna(inplace=True)
    features = calculate_features(data)
    res = model.predict(features)

    # for prediction in res:
    #     print(prediction)

    data = data[: len(features)]

    print(len(res))
    print(len(features))
    print(len(data))

    data["label"] = ""
    print(data)

    # 0 means walking, 1 means jumping: replace these values with the proper labels
    data["label"] = ["walking" if x == 0 else "jumping" for x in res]

    # assign labels to first len(res) rows of data
    # data.iloc[:len(res), -1] = ["walking" if x == 0.0 else "jumping" for x in res]
    return data
