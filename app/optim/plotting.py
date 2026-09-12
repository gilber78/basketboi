"""
BASKETBOI

Copyright © 2026 Eric Gilbertson. All rights reserved.
See LICENSE.md for permitted use.
"""

import numpy as np
import matplotlib.pyplot as plt

import statistics as stats


def plot_2d_histogram(x, y, title, binwidth=1, xlabel="Predicted values", ylabel="True values"):
    """
    Plot a 2d histogram of a dataset x and y
    """
    plt.figure()
    plt.title(title, wrap=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(y, y, "w", alpha=0.6)
    plt.hist2d(x, y, bins=[int(max(x) - min(x) / binwidth) + 1, int(max(y) - min(y) / binwidth) + 1])


def plot_pdf_function(x, y, title, binwidth=0.05, bounds=(0, 1), xlabel="Predicted Probability", ylabel="True Probability", std=None):
    """
    Plot probability distribution function for a dataset x and y based on supplied bin sizes
    """
    xvals, yvals, m, b = stats.calc_calibrated_slope_intercept(x, y, binwidth, bounds, True)
    liney = m * xvals + b
    plt.figure()
    plt.title(title, wrap=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(bounds)
    plt.ylim((0, 1))
    plt.scatter(xvals, yvals, alpha=1)
    plt.plot(xvals, liney, alpha=0.6)
    # this isn't super useful for the win model, but a good-to-have for everything else
    if std is not None:
        plt.plot(xvals, liney + std, "g", alpha=0.75)
        plt.plot(xvals, liney - std, "g", alpha=0.75)
    plt.plot(xvals, xvals + 0.05, "k", alpha=0.24)
    plt.plot(xvals, xvals - 0.05, "k", alpha=0.24)
    plt.plot(xvals, xvals, "k", alpha=0.6)
    plt.legend([f"m = {m}", f"b = {b}", f"std = {std}"])


def plot_ROC_curve(x, y, title, binwidth=0.01, bounds=(0, 1), xlabel="FPR", ylabel="TPR"):
    """
    Plot receiver operating characteristic curve
    """
    TPR, FPR, _ = stats.calc_ROC_curve(x, y, binwidth, bounds)
    plt.figure()
    plt.title(title, wrap=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(bounds)
    plt.ylim((0, 1))
    plt.plot(FPR, TPR, "g", alpha=1)
    plt.plot(FPR, FPR, "k", alpha=0.6)


def plot_pdf_function_DEBUG(x, y, title, binwidth, bounds, xlabel="Input Term", ylabel="Output Model Value"):
    """
    Plot pdf function in tandem with linear and cubic fits. For debug/model tuning only
    """
    bins = np.linspace(bounds[0], bounds[1], int((bounds[1] - bounds[0]) / binwidth) + 1)
    xvals = [(bins[i] + bins[i - 1]) / 2 for i in range(1, len(bins))]
    yvals = []
    for i in range(1, len(bins)):
        y_mask = y[(bins[i - 1] <= x) & (x <= bins[i])]
        if len(y_mask) == 0:
            yvals.append(0)  # this is 0 so that we don't plot data we don't use
        else:
            yvals.append(sum(y_mask) / len(y_mask))
    xvals = np.array(xvals)
    yvals = np.array(yvals)

    # SPECIFIC TO DEBUG ONLY!! ::: drop entries below <0.05 and >0.95 PoD mask
    inds_to_drop = np.where((yvals < 0.05) | (yvals > 0.95))[0]
    xvals = np.delete(xvals, inds_to_drop)
    yvals = np.delete(yvals, inds_to_drop)

    # regressions for degree 1 and 3 and respective curves
    m, b = np.polyfit(xvals, yvals, 1)
    k3, k2, k1, k0 = np.polyfit(xvals, yvals, 3)
    liney = m * xvals + b
    cubey = k3 * xvals**3 + k2 * xvals**2 + k1 * xvals + k0

    # calculate and return debug-debug fitness score(s) r2 * sin^2(2 * arctan(m))
    xscaled = (xvals - np.min(x)) / (np.max(x) - np.min(x))
    line_slope = (np.max(liney) - np.min(liney)) / (xscaled[np.argmax(liney)] - xscaled[np.argmin(liney)])
    cube_slope = (np.max(cubey) - np.min(cubey)) / (xscaled[np.argmax(cubey)] - xscaled[np.argmin(cubey)])
    liner2 = 1 - np.sum((yvals - liney) ** 2) / np.sum((yvals - np.mean(yvals)) ** 2)
    cuber2 = 1 - np.sum((yvals - cubey) ** 2) / np.sum((yvals - np.mean(yvals)) ** 2)
    line_strength_of_signal = liner2 * np.sin(2 * np.atan(line_slope)) ** 2
    cube_strength_of_signal = cuber2 * np.sin(2 * np.atan(cube_slope)) ** 2
    print(line_strength_of_signal, cube_strength_of_signal)

    # plotting functionality
    plt.figure()
    plt.title(title, wrap=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim((0, 1))
    plt.scatter(xvals, yvals, alpha=1)
    plt.plot(xvals, liney, "r")
    plt.plot(xvals, cubey, "g")
    plt.legend(["data points", f"line = {np.round(liner2, 5)}", f"cube = {np.round(cuber2, 5)}"])
