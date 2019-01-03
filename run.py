# -*- coding: utf-8 -*-
"""
Runs the training for our best model.

The best model was:
Least squares, using offset, standardization, and outlier removal.
The training is performed once per category (each type of jet).

The predictions are saved for submission in "best_model.csv".

"""

from proj1_helpers import *
from custom_helpers import *
from implementations import *
import numpy as np


def train_model(y_train, x_train, flag_method, max_iter=1000, gamma=0.001, lambda_=0.0):
    """Train a model on a given subset of the data. Choose method by setting flag_method"""

    initial_w = np.ones(x_train.shape[1])

    if flag_method == 0:
        # Use linear regression (full gradient descent)
        weight, loss_tr = least_squares_GD(
            y_train, x_train, initial_w, max_iters, gamma)

    if flag_method == 1:
        # Use linear regression (stochastic gradient descent)
        weight, loss_tr = least_squares_SGD(
            y_train, x_train, initial_w, max_iters, gamma)

    if flag_method == 2:
        # Use least squares method
        weight, loss_tr = least_squares(y_train, x_train)

    if flag_method == 3:
        # Use ridge regression
        weight, loss_tr = ridge_regression(y_train, x_train, lambda_)

    if flag_method == 4:
        # Use logistic regression
        weight, loss_tr = logistic_regression(
            y_train, x_train, initial_w, max_iters, gamma)

    if flag_method == 5:
        # Use regularized logistic regression
        weight, loss_tr = reg_logistic_regression(
            y_train, x_train, initial_w, max_iters, gamma, lambda_)

    return weight


# Load Data
print("Loading Data, please wait")
y_test, x_test_raw, ids_test = load_csv_data('data/test.csv')
y_train, x_train_raw, ids_train = load_csv_data('data/train.csv')
print("Data loaded")


"""
Methods mapping
0    Linear Regression (Full gradient descent)
1    Linear Regression (Stochastic gradient descent)
2    Least Squares Method
3    Ridge Regression
4    Logistic Regression (Stochastic Gradient Descent)
5    Regularized Logistic Regression (Stochastic Gradient Descent)
"""

# Choose feature treatment methods
flag_add_offset = True
flag_standardize = True
flag_remove_outliers = True
degree = 2

# Choose training model to apply (see mapping above)
flag_method = 2

# Set training parameters
max_iters = 5000
gamma = 0.01
lambda_ = 0.0


# In the dateset, we found that the Column[22] PRI_jet_num dataset is categorical with Four categories defined.column_jet_nb = 22
pred_y = []
ids_pred_y = []
column_categorical = 22
# In the paper where describes the differente features of the data, explains that different columns are invalid values
# depending on the value of the categorical feature, so we can delete those values for the 4 different trainings
# The undefined features, with first vector for the categorical value of 0, and so on.
undefined_features = [[4, 5, 6, 12, 22, 23, 24, 25, 26,
                       27, 28, 29], [4, 5, 6, 12, 22, 26, 27, 28], [22], [22]]

# We will have a for loop with 4 values for the 4 categorical training
for nb_jets in range(0, 4):
    print("Cleaning and preparing data for jet number %d" % nb_jets)
    # We will separate select data according to the value of the categorical values for each loop in our cicle
    jet_index_test = x_test_raw[:, column_categorical] == nb_jets
    x_test_jet = x_test_raw[jet_index_test]
    y_test_jet = y_test[jet_index_test]
    id_test_jet = ids_test[jet_index_test]
    jet_index_train = x_train_raw[:, column_categorical] == nb_jets
    x_train_jet = x_train_raw[jet_index_train]
    y_train_jet = y_train[jet_index_train]
    id_train_jet = ids_train[jet_index_train]

    # remove undefined features
    x_test_jet = np.delete(x_test_jet, undefined_features[nb_jets], axis=1)
    x_train_jet = np.delete(x_train_jet, undefined_features[nb_jets], axis=1)

    # Prepare data
    x_train_jet, x_test_jet = prepare_data(
        x_train_jet, x_test_jet, flag_add_offset, flag_standardize, flag_remove_outliers, degree)

    print("Training model for jet number %d..." % nb_jets)
    # train the chosen model
    weight = train_model(y_train_jet, x_train_jet,
                         flag_method, max_iters, gamma, lambda_)

    print("making predictions for jet number %d..." % nb_jets)
    # Now we get the prediction for y
    pred_y_jet = predict_labels(weight, x_test_jet)
    pred_y.extend(pred_y_jet)
    ids_pred_y.extend(id_test_jet)

print("Finished training all four models.")

# Choose filename:
filename = "best_model.csv"

print("Creating submission file.")
create_csv_submission(ids_pred_y, pred_y, filename)
print("Created submission file!")
