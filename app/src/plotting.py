import numpy as np
import matplotlib.pyplot as plt

import statistics as stats


def plot_2d_histogram(x, y, title, binwidth=1, xlabel="Predicted values", ylabel="True values"):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(y, y, "w", alpha=0.6)
    plt.hist2d(x, y, bins=[int(max(x) - min(x) / binwidth) + 1, int(max(y) - min(y) / binwidth) + 1])


def plot_pdf_function(x, y, title, binwidth=0.05, bounds=(0, 1), xlabel="Predicted Probability", ylabel="True Probability"):
    xvals, yvals, m, b = stats.calc_calibrated_slope_intercept(x, y, binwidth, bounds, True)
    liney = m * xvals + b
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(bounds)
    plt.ylim((0, 1))
    plt.scatter(xvals, yvals, alpha=1)
    plt.plot(xvals, liney, alpha=0.6)
    plt.plot(xvals, xvals + 0.05, "k", alpha=0.24)
    plt.plot(xvals, xvals - 0.05, "k", alpha=0.24)
    plt.plot(xvals, xvals, "k", alpha=0.6)
    plt.legend([f"m = {m}", f"b = {b}"])


def plot_ROC_curve(x, y, title, binwidth=0.01, bounds=(0, 1), xlabel="FPR", ylabel="TPR"):
    TPR, FPR, _ = stats.calc_ROC_curve(x, y, binwidth, bounds)
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(bounds)
    plt.ylim((0, 1))
    plt.plot(FPR, TPR, "g", alpha=1)
    plt.plot(FPR, FPR, "k", alpha=0.6)
