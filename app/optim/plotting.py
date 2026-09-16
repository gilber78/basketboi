"""
BASKETBOI

Copyright © 2026 Eric Gilbertson. All rights reserved.
See LICENSE.md for permitted use.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot

import statistics as stats


def plot_pdf_function(x, y, title, binwidth=0.05, bounds=(0, 1), xlabel="Predicted Probability", ylabel="True Probability", std=None):
    """
    Plot probability distribution function for a dataset x and y based on supplied bin sizes
    """
    xvals, yvals, m, b = stats.calc_calibrated_slope_intercept_binned(x, y, binwidth, bounds, True)
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


def plot_2d_histogram(x, y, title, binwidth=1, xlabel="Predicted values", ylabel="True values", std=None):
    """
    Plot a 2d histogram of a dataset x and y
    """
    m, b = np.polyfit(x, y, 1)
    xsorted_unique = np.unique(np.sort(x))
    liney = m * xsorted_unique + b
    plt.figure()
    plt.title(title, wrap=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist2d(x, y, bins=[int(max(x) - min(x) / binwidth) + 1, int(max(y) - min(y) / binwidth) + 1])
    plt.plot(xsorted_unique, liney, "m", alpha=0.6)
    if std is not None:
        plt.plot(xsorted_unique, liney + std, "c", alpha=0.75)
        plt.plot(xsorted_unique, liney - std, "c", alpha=0.75)
    plt.plot(xsorted_unique, xsorted_unique, "w", alpha=0.6)
    plt.legend([f"m = {m}", f"b = {b}", f"std = {std}"])


def plot_1d_histogram_subplots(x, y, title, binwidth=1, sgxtitle="Predicted Value", sgytitle="True Value", sgxytitle="Delta Value", std=None):
    """
    Plot a system of histograms that show x, y-x, and y series
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(title, wrap=True)
    ax[0].set_title(sgxtitle)
    ax[0].hist(x, bins=int(max(x) - min(x) / binwidth) + 1)
    ax[1].set_title(sgxytitle)
    ax[1].hist(y - x, bins=int(max(y - x) - min(y - x) / binwidth) + 1)
    ax[2].set_title(sgytitle)
    ax[2].hist(y, bins=int(max(y) - min(y) / binwidth) + 1)
    if std is not None:
        mult = len(x)
        xsorted_unique = np.unique(np.sort(x))
        delsorted_unique = np.unique(np.sort(y - x))
        ysorted_unique = np.unique(np.sort(y))
        ax[0].plot(np.mean(x) - std, 0, "kx")
        ax[0].plot(np.mean(x) + std, 0, "kx")
        ax[0].plot(xsorted_unique, mult * norm.pdf(xsorted_unique, loc=np.mean(x), scale=np.std(x)), "k", alpha=0.6)
        ax[1].plot(np.mean(y - x) - std, 0, "kx")
        ax[1].plot(np.mean(y - x) + std, 0, "kx")
        ax[1].plot(delsorted_unique, mult * norm.pdf(delsorted_unique, loc=0, scale=std), "b", alpha=0.75)
        ax[1].plot(delsorted_unique, mult * norm.pdf(delsorted_unique, loc=np.mean(y - x), scale=np.std(y - x)), "k", alpha=0.6)
        ax[2].plot(np.mean(y) - std, 0, "kx")
        ax[2].plot(np.mean(y) + std, 0, "kx")
        ax[2].plot(ysorted_unique, mult * norm.pdf(ysorted_unique, loc=np.mean(y), scale=np.std(y)), "k", alpha=0.6)


def plot_qqs(x, y, title, sgxtitle="QQ of Predicted Values", sgytitle="QQ of True Values", sgxytitle="QQ of Residuals"):
    """
    Generate Quartile-Quartile plots for continuous xy data
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(title, wrap=True)
    probplot(x, dist="norm", plot=ax[0])
    probplot(y - x, dist="norm", plot=ax[1])
    probplot(y, dist="norm", plot=ax[2])
    ax[0].set_title(sgxtitle)
    ax[1].set_title(sgxytitle)
    ax[2].set_title(sgytitle)


def plot_pdf_function_DEBUG(x, y, title, binwidth, bounds, xlabel="Input Term", ylabel="Output Model Value", sos_mult=1):
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
    line_strength_of_signal = sos_mult * liner2 * np.sin(2 * np.atan(line_slope)) ** 2
    cube_strength_of_signal = sos_mult * cuber2 * np.sin(2 * np.atan(cube_slope)) ** 2
    print("    All Outcomes:   ", np.round(line_strength_of_signal, 4), np.round(cube_strength_of_signal, 4))

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


def plot_scatterplot_subplots_DEBUG(x, y, xwin, ywin, xlose, ylose, title, xlabel="Input Term", ylabel="Output Model Value", sos_mult=1):
    """
    Plot a system of scatterplots that shows combined, home win, and home lose situations for the term
    """
    # regressions for degree 1 and 3 and respective curves
    m, b = np.polyfit(x, y, 1)
    k3, k2, k1, k0 = np.polyfit(x, y, 3)
    sind = np.argsort(x)
    xsorted, ysorted = x[sind], y[sind]
    xsorted_unique = np.unique(x[sind])
    liney = m * xsorted + b
    liney_unique = m * xsorted_unique + b
    cubey = k3 * xsorted**3 + k2 * xsorted**2 + k1 * xsorted + k0
    cubey_unique = k3 * xsorted_unique**3 + k2 * xsorted_unique**2 + k1 * xsorted_unique + k0
    m, b = np.polyfit(xwin, ywin, 1)
    k3, k2, k1, k0 = np.polyfit(xwin, ywin, 3)
    sind = np.argsort(xwin)
    xwinsorted, ywinsorted = xwin[sind], ywin[sind]
    xwinsorted_unique = np.unique(xwin[sind])
    winliney = m * xwinsorted + b
    winliney_unique = m * xwinsorted_unique + b
    wincubey = k3 * xwinsorted**3 + k2 * xwinsorted**2 + k1 * xwinsorted + k0
    wincubey_unique = k3 * xwinsorted_unique**3 + k2 * xwinsorted_unique**2 + k1 * xwinsorted_unique + k0
    m, b = np.polyfit(xlose, ylose, 1)
    k3, k2, k1, k0 = np.polyfit(xlose, ylose, 3)
    sind = np.argsort(xlose)
    xlosesorted, ylosesorted = xlose[sind], ylose[sind]
    xlosesorted_unique = np.unique(xlose[sind])
    loseliney = m * xlosesorted + b
    loseliney_unique = m * xlosesorted_unique + b
    losecubey = k3 * xlosesorted**3 + k2 * xlosesorted**2 + k1 * xlosesorted + k0
    losecubey_unique = k3 * xlosesorted_unique**3 + k2 * xlosesorted_unique**2 + k1 * xlosesorted_unique + k0

    # calculate r2 values
    liner2 = 1 - np.sum((ysorted - liney) ** 2) / np.sum((ysorted - np.mean(ysorted)) ** 2)
    cuber2 = 1 - np.sum((ysorted - cubey) ** 2) / np.sum((ysorted - np.mean(ysorted)) ** 2)
    winliner2 = 1 - np.sum((ywinsorted - winliney) ** 2) / np.sum((ywinsorted - np.mean(ywinsorted)) ** 2)
    wincuber2 = 1 - np.sum((ywinsorted - wincubey) ** 2) / np.sum((ywinsorted - np.mean(ywinsorted)) ** 2)
    loseliner2 = 1 - np.sum((ylosesorted - loseliney) ** 2) / np.sum((ylosesorted - np.mean(ylosesorted)) ** 2)
    losecuber2 = 1 - np.sum((ylosesorted - losecubey) ** 2) / np.sum((ylosesorted - np.mean(ylosesorted)) ** 2)

    # calculate and return debug-debug fitness score(s) r2 * sin^2(2 * arctan(m))
    xscaled = (xsorted_unique - np.min(x)) / (np.max(x) - np.min(x))
    lineyscaled = (liney_unique - np.min(y)) / (np.max(y) - np.min(y))
    cubeyscaled = (cubey_unique - np.min(y)) / (np.max(y) - np.min(y))
    line_slope = (np.max(lineyscaled) - np.min(lineyscaled)) / (xscaled[np.argmax(lineyscaled)] - xscaled[np.argmin(lineyscaled)])
    cube_slope = (np.max(cubeyscaled) - np.min(cubeyscaled)) / (xscaled[np.argmax(cubeyscaled)] - xscaled[np.argmin(cubeyscaled)])
    line_strength_of_signal = sos_mult * liner2 * np.sin(2 * np.atan(line_slope)) ** 2
    cube_strength_of_signal = sos_mult * cuber2 * np.sin(2 * np.atan(cube_slope)) ** 2
    xwinscaled = (xwinsorted_unique - np.min(x)) / (np.max(x) - np.min(x))
    winlineyscaled = (winliney_unique - np.min(y)) / (np.max(y) - np.min(y))
    wincubeyscaled = (wincubey_unique - np.min(y)) / (np.max(y) - np.min(y))
    winline_slope = (np.max(winlineyscaled) - np.min(winlineyscaled)) / (
        xwinscaled[np.argmax(winlineyscaled)] - xwinscaled[np.argmin(winlineyscaled)]
    )
    wincube_slope = (np.max(wincubeyscaled) - np.min(wincubeyscaled)) / (
        xwinscaled[np.argmax(wincubeyscaled)] - xwinscaled[np.argmin(wincubeyscaled)]
    )
    winline_strength_of_signal = sos_mult * winliner2 * np.sin(2 * np.atan(winline_slope)) ** 2
    wincube_strength_of_signal = sos_mult * wincuber2 * np.sin(2 * np.atan(wincube_slope)) ** 2
    xlosescaled = (xlosesorted_unique - np.min(x)) / (np.max(x) - np.min(x))
    loselineyscaled = (loseliney_unique - np.min(y)) / (np.max(y) - np.min(y))
    losecubeyscaled = (losecubey_unique - np.min(y)) / (np.max(y) - np.min(y))
    loseline_slope = (np.max(loselineyscaled) - np.min(loselineyscaled)) / (
        xlosescaled[np.argmax(loselineyscaled)] - xlosescaled[np.argmin(loselineyscaled)]
    )
    losecube_slope = (np.max(losecubeyscaled) - np.min(losecubeyscaled)) / (
        xlosescaled[np.argmax(losecubeyscaled)] - xlosescaled[np.argmin(losecubeyscaled)]
    )
    loseline_strength_of_signal = sos_mult * loseliner2 * np.sin(2 * np.atan(loseline_slope)) ** 2
    losecube_strength_of_signal = sos_mult * losecuber2 * np.sin(2 * np.atan(losecube_slope)) ** 2
    print("    Home Team Wins: ", np.round(winline_strength_of_signal, 4), np.round(wincube_strength_of_signal, 4))
    print("    All Outcomes:   ", np.round(line_strength_of_signal, 4), np.round(cube_strength_of_signal, 4))
    print("    Home Team Loses:", np.round(loseline_strength_of_signal, 4), np.round(losecube_strength_of_signal, 4))

    # plotting functionality
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(title, wrap=True)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title("Home Team Wins")
    ax[0].scatter(xwin, ywin, alpha=0.25)
    ax[0].plot(xwinsorted, winliney, "r")
    ax[0].plot(xwinsorted, wincubey, "g")
    ax[0].legend(["data points", f"line = {np.round(winliner2, 5)}", f"cube = {np.round(wincuber2, 5)}"])
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    ax[1].set_title("All Outcomes")
    ax[1].scatter(x, y, alpha=0.25)
    ax[1].plot(xsorted, liney, "r")
    ax[1].plot(xsorted, cubey, "g")
    ax[1].legend(["data points", f"line = {np.round(liner2, 5)}", f"cube = {np.round(cuber2, 5)}"])
    ax[2].set_xlabel(xlabel)
    ax[2].set_ylabel(ylabel)
    ax[2].set_title("Home Team Loses")
    ax[2].scatter(xlose, ylose, alpha=0.25)
    ax[2].plot(xlosesorted, loseliney, "r")
    ax[2].plot(xlosesorted, losecubey, "g")
    ax[2].legend(["data points", f"line = {np.round(loseliner2, 5)}", f"cube = {np.round(losecuber2, 5)}"])
