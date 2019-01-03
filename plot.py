# -*- coding: utf-8 -*-
"""plot helpers for cross validation."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization_lambda(lambdas, mse_tr, mse_te):
    """Visualization of the curves of mse_tr and mse_te for lambda.

    Args:
        lambdas (numpy.array): Lambda values
        mse_tr (numpy.array): MSE Training 
        mse_te (numpy.array): MSE Test
    """

    plt.plot(lambdas, mse_tr, marker="x", color='b', label='train error')
    plt.plot(lambdas, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=0)
    plt.grid(True)
    plt.savefig("cross_validation_lambda", bbox_inches="tight")


def cross_validation_visualization_degree(degrees, mse_tr, mse_te):
    """Visualization of the curves of mse_tr and mse_te for the degrees.

    Args:
        degrees (numpy.array): Degree values
        mse_tr (numpy.array): MSE Training 
        mse_te (numpy.array): MSE Test
    """

    plt.plot(degrees, mse_tr, marker="x", color='b', label='train error')
    plt.plot(degrees, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("degree")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=0)
    plt.grid(True)
    plt.ylim((0.6, 1))
    plt.xticks(degrees)
    plt.savefig("cross_validation_degree", bbox_inches="tight")
