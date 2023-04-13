import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from data_setup import setup_data

######################## FEATURE EXTRACTION ########################
def calculate_features(data):

    # will return dataframe with each acc dimension's features
    features = pd.DataFrame()
   
    column_names = [
        ["max_X", "max_Y", "max_Z"],
        ["min_X", "min_Y", "min_Z"],
        ["range_X", "range_Y", "range_Z"],
        [
            "mean_X",
            "mean_Y",
            "mean_Z",
        ],
        ["median_X", "median_Y", "median_Z"],
        ["var_X", "var_Y", "var_Z"],
        ["skew_X", "skew_Y", "skew_Z"],
        ["kurtosis_X", "kurtosis_Y", "kurtosis_Z"],
        ["deviation_X", "deviation_Y", "deviation_Z"],
        ["quantile_X", "quantile_Y", "quantile_Z"],
    ]

    # 500 samples: 5 second intervals @ 100 Hz sampling rate, which is what Phyphox uses
    # split into 500 sample groups and find the features
    feature_vals = []
    feature_vals.append(data.rolling(window=500).max())
    feature_vals.append(data.rolling(window=500).min())
    feature_vals.append(feature_vals[0] - feature_vals[1])
    feature_vals.append(data.rolling(window=500).mean())
    feature_vals.append(data.rolling(window=500).median())
    feature_vals.append(data.rolling(window=500).var())
    feature_vals.append(data.rolling(window=500).skew())
    feature_vals.append(data.rolling(window=500).kurt())
    feature_vals.append(data.rolling(window=500).std())
    feature_vals.append(data.rolling(window=500).quantile(0.5))

    # then, populate dataframe with features split into X Y and Z format using the column names list and calculated features
    for i in range(len(feature_vals)):
        features[column_names[i][0]] = feature_vals[i].iloc[:, 0]
        features[column_names[i][1]] = feature_vals[i].iloc[:, 1]
        features[column_names[i][2]] = feature_vals[i].iloc[:, 2]

    # drop none values (size will dictate size of result)
    features.dropna(inplace=True)
    return features


# used in app.py
def train():
    # get dataframes
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

    # label columns and drop unnecessary ones
    data.columns = ["time", "x-g", "y-g", "z-g", "abs-g"]
    data.drop(["time", "abs-g"], axis=1, inplace=True)

    # normalize data and get features: then predict using passed in model
    data = data.rolling(25, min_periods=1).mean()
    features = calculate_features(data)
    res = model.predict(features)

    # ignore number of NA vals that were dropped in feature extraction
    data = data[: len(features)]

    # 0 means walking, 1 means jumping: replace these values with the proper labels
    data["label"] = ["walking" if x == 0 else "jumping" for x in res]
    return data
