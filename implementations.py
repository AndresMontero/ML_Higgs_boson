# -*- coding: utf-8 -*-
"""Contains the 6 ML methods that were required to be implemented."""
import csv
import numpy as np

# ----------------------- Least Squares using Gradient Descent ---------------------------


def compute_mse(y, tx, w):
    """Compute the loss by mse

    Args:
        y (numpy.array): y values
        tx (numpy.array): Transposed x values
        w (numpy.array): Weight

    Returns:
        numpy.array: The calculated MSE
    """

    pred = np.squeeze(tx.dot(w))
    e = y - pred
    mse = e.dot(e) / (2 * len(e))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient

    Args:
        y (numpy.array): y values
        tx (numpy.array): transposed x values
        w (numpy.array): weight

    Returns:
        numpy.array: The calculated gradient.
    """

    err = y - tx.dot(w)
    gradient = -tx.T.dot(err) / len(err)
    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm

    Args:
        y (numpy.array): y values
        tx (numpy.array): transposed x values
        initial_w (numpy.array): initial weight.
        max_iters (int): number of iterations.
        gamma(float): the gamma to use.

    Returns:
        (tuple): tuple containing:

            w (numpy.array): Weight result
            loss (numpy.array): Loss result
    """
    w = initial_w
    print_every = 50
    cumulative_loss = 0

    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * gradient
        # compute loss
        loss = compute_mse(y, tx, w)
        cumulative_loss += loss

        if (n_iter % print_every == 0):
            # print average loss for the last print_every iterations
            print('iteration\t', str(n_iter), '\tloss: ',
                  str(cumulative_loss / print_every))
            cumulative_loss = 0

    return w, loss


# ----------------------- Least Squares using Stochastic Gradient Descent ---------------------------
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Args:
        y (numpy.array): y values
        tx (numpy.array): Transposed x values
        batch_size (int): Size of the batch
        num_batches (int, optional): Defaults to 1. Number of batches
        shuffle (bool, optional): Defaults to True. Shuffle or not
    """

    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic descent algorithm

    Args:
        y (numpy.array): y values
        tx (numpy.array): transposed x values
        initial_w (numpy.array): initial weight.
        max_iters (int): number of iterations.
        gamma(float): the gamma to use.

    Returns:
        (tuple): tuple containing:

            w (numpy.array): Weight result
            loss (numpy.array): Loss result
    """

    # Define parameters to store w and loss
    w = initial_w
    batch_size = 1
    print_every = 50
    cumulative_loss = 0

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse(y, tx, w)
            cumulative_loss += loss

        if (n_iter % print_every == 0):
            # print average loss for the last print_every iterations
            print('iteration\t', str(n_iter), '\tloss: ',
                  str(cumulative_loss / print_every))
            cumulative_loss = 0

    return w, loss


# ----------------------- Least Squares ---------------------------
def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y (numpy.array): y values
        tx (numpy.array): transposed x values

    Returns:
        (tuple): tuple containing:

            w (numpy.array): Weight result
            loss (numpy.array): Loss result
    """

    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = np.sqrt(2 * compute_mse(y, tx, w))
    return w, loss


# ----------------------- Ridge Regression ---------------------------
def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Args:
        y (numpy.array): y values
        tx (numpy.array): transposed x values
        lambda_ (float): the lambda value to use.

    Returns:
        (tuple): tuple containing:

            w (numpy.array): Weight result
            loss (numpy.array): Loss result
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = np.sqrt(2 * compute_mse(y, tx, w))
    return w, loss


# ----------------------- Logistic Regression ---------------------------
def sigmoid(t):
    """Apply sigmoid function on t

    Args:
        t (numpy.array): Value to use

    Returns:
        numpy.array: Calculated sigmoid
    """

    return 1.0 / (1.0 + np.exp(-t))


def calculate_log_loss(y, tx, w):
    """Compute the cost by negative log likelihood

    Args:
        y (numpy.array): y values
        tx (numpy.array): Transposed x values
        w (numpy.array): Weight

    Returns:
        numpy.array: Calculated logistic loss
    """

    xtw = (tx.dot(w))
    loss = np.sum(np.log(1 + np.exp(xtw))) - y.T.dot(xtw)

    return loss / tx.shape[0]


def calculate_log_gradient(y, tx, w):
    """Compute the gradient of loss.

    Args:
        y (numpy.array): y values
        tx (numpy.array): Transposed x values
        w (numpy.array): Weight

    Returns:
        numpy.array: Calculated logistic gradient
    """

    sig = sigmoid(tx.dot(w))
    gradient = tx.T.dot(sig - y)

    return gradient / tx.shape[0]


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Implement logistic regression using full gradient descent

    Args:
        y (numpy.array): y values
        tx (numpy.array): transposed x values
        initial_w (numpy.array): initial weight.
        max_iters (int): number of iterations.
        gamma(float): the gamma to use.

    Returns:
        (tuple): tuple containing:

            w (numpy.array): Weight result
            loss (numpy.array): Loss result
    """

    print_every = 50
    cumulative_loss = 0
    w = initial_w

    # change labels from (-1, 1) to (0, 1)
    y[np.where(y == -1)] = 0

    for n_iter in range(max_iters):

        gradient = calculate_log_gradient(y, tx, w)
        loss = calculate_log_loss(y, tx, w)
        cumulative_loss += loss
        w = w - (gamma * gradient)

        if (n_iter % print_every == 0):
            # print average loss for the last print_every iterations
            print('iteration\t', str(n_iter), '\tloss: ',
                  str(cumulative_loss / print_every))
            cumulative_loss = 0

    return w, loss


# ----------------------- Regularized Logistic Regression ---------------------------
def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    """Implement regularized logistic regression using full gradient descent

    Args:
        y (numpy.array): y values
        tx (numpy.array): transposed x values
        initial_w (numpy.array): initial weight.
        max_iters (int): number of iterations.
        gamma(float): the gamma to use.
        lambda(float): the lambda to use.

    Returns:
        (tuple): tuple containing:

            w (numpy.array): Weight result
            loss (numpy.array): Loss result
    """

    print_every = 50
    cumulative_loss = 0
    w = initial_w

    # change labels from (-1, 1) to (0, 1)
    y[np.where(y == -1)] = 0

    for n_iter in range(max_iters):
        loss = calculate_log_loss(
            y, tx, w) + ((lambda_ / 2) * (np.linalg.norm(w) ** 2)) / tx.shape[0]
        grad = calculate_log_gradient(
            y, tx, w) + (lambda_ * np.sum(w)) / tx.shape[0]
        w = w - gamma * grad
        cumulative_loss += loss

        if (n_iter % print_every == 0):
            # print average loss for the last print_every iterations
            print('iteration\t', str(n_iter), '\tloss: ',
                  str(cumulative_loss / print_every))
            cumulative_loss = 0
    return w, loss
