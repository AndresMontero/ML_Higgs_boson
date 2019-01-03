# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from implementations import *


def standardize(x):
    """Standardize the original data set.

    Args:
        x (numpy.array): Array with data for x

    Returns:
        (tuple): tuple containing:

            x (numpy.array): Standardized array
            mean_x (numpy.array): Arithmetic Mean
            std_x (numpy.array): Standard deviation
    """

    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def standardize_test(x_test, mean, std):
    """Standardize the values of testing_x depending on the values of mean and std of the training x vector

    Args:
        x_test (numpy.array): Testing x values
        mean (numpy.array): Mean values
        std (numpy.array): Standard deviation values

    Returns:
        numpy.array: Standarized X
    """

    new_x = x_test.copy()
    new_x = (new_x - mean) / std
    return new_x


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x (numpy.array): Data for x
        degree (int): Degree for the polynomial

    Returns:
        numpy.array: Generated Polynomial
    """

    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def remove_invalid(x):
    """Replaces the invalid values -999 by the mean of the entire column

    Args:
        x (numpy.array): Values for x

    Returns:
        numpy.array: New values without invalid entries. 
    """

    x_new = []
    # first replace the values of -999 with nan
    x_mod = np.where(x == -999, np.nan, x)
    # then obtain the mean of the column ignoring the nan values
    mean = np.nanmean(x_mod, axis=0)
    # now we search the -999 in the original data and replace it with the mean of the column
    for i in range(x.shape[1]):
        x_new.append(np.where(x[:, i] == -999, mean[i], x[:, i]))
    return np.array(x_new).T


def remove_outliers(x):
    """Removes with IQR method, multiplying the IQR by 1.5

    Args:
        x (numpy.array): Values for x

    Returns:
        numpy.array: New values with cleaned data
    """

    data_clean = np.zeros((x.shape))
    for i in range(x.shape[1]):
        col = x[:, i]
        data_copy = np.array(col)
        counts = np.bincount(np.absolute(data_copy.astype(int)))
        replace_most_frecuent = np.argmax(counts)
        upper_quartile = np.percentile(data_copy, 75)
        lower_quartile = np.percentile(data_copy, 25)
        IQR = upper_quartile - lower_quartile
        valid_data = (lower_quartile - IQR*1.5, upper_quartile + IQR*1.5)
        j = 0
        for y in data_copy.tolist():
            if y >= valid_data[0] and y <= valid_data[1]:
                data_clean[j][i] = y
                j = j + 1
            else:
                data_clean[j][i] = replace_most_frecuent
                j = j + 1
    return data_clean


def prepare_data(x_train, x_test, flag_add_offset, flag_standardize, flag_remove_outliers, degree):
    """Prepare data. Different manipulations can be specified with flags

    Args:
        x_train (numpy.array): Values of x for training
        x_test (numpy.array): Values of x for testing
        flag_add_offset (bool): Flag for adding offset value
        flag_standardize (bool): Flag for standardizing
        flag_remove_outliers (bool): Flag for removing outliers
        degree (int): Degree for polynomial base

    Returns:
        (tuple): tuple containing:

            x_train (numpy.array): New values of x for training 
            x_test (numpy.array): New values of x for testing
    """

    # remove invalid values (-999)
    x_train = remove_invalid(x_train)
    x_test = remove_invalid(x_test)

    if flag_remove_outliers == True:
        # replace the outliers with the most common element of each column
        x_train = remove_outliers(x_train)
        x_test = remove_outliers(x_test)

    # Building Polynomial base with degree passed
    x_train = build_poly(x_train, degree)
    x_test = build_poly(x_test, degree)

    if flag_standardize == True:
        # Standardizing data
        x_train, mean, std = standardize(x_train)
        x_test = standardize_test(x_test, mean, std)

    if flag_add_offset == True:
        # Getting matrix tX, adding offset value, entire colum of ones[1]
        x_train = np.c_[np.ones(x_train.shape[0]), x_train]
        x_test = np.c_[np.ones(x_test.shape[0]), x_test]

    return x_train, x_test


def predict_labels_logistic(weights, data):
    """Generates class predictions given weights, and a test data matrix.
    Notice that for logistic regression the values are within 0 and 1, so decission limits are changed

    Args:
        weights (numpy.array): Weights to be used
        data (numpy.array): Test data

    Returns:
        numpy.array: Predictions generated
    """

    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred
