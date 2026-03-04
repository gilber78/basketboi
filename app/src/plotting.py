import numpy as np
import matplotlib.pyplot as plt


def plot_2d_histogram(x, y, title, binwidth=1, xlabel="Predicted values", ylabel="True values"):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(y, y, "w", alpha=0.6)
    plt.hist2d(x, y, bins=[int(max(x) - min(x) / binwidth) + 1, int(max(y) - min(y) / binwidth) + 1])


def plot_pdf_function(x, y, title, binwidth=0.05, bounds=(0, 1), xlabel="Predicted Probability", ylabel="True Probability"):
    bins = np.linspace(bounds[0], bounds[1], int((bounds[1] - bounds[0]) / binwidth) + 1)
    xvals = [(bins[i] + bins[i - 1]) / 2 for i in range(1, len(bins))]
    yvals = []
    weights = []
    for i in range(1, len(bins)):
        y_mask = y[(bins[i - 1] <= x) & (x <= bins[i])]
        yvals.append(sum(y_mask) / len(y_mask))
        weights.append(len(y_mask))
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    weights = np.array(weights)

    # TODO delete this block after troubleshooting first model
    m, b = np.polyfit(xvals, yvals, 1, w=weights)
    a2, a1, a0 = np.polyfit(xvals, yvals, 2, w=weights)
    k3, k2, k1, k0 = np.polyfit(xvals, yvals, 3, w=weights)
    liney = m * xvals + b
    quady = a2 * xvals**2 + a1 * xvals + a0
    cubey = k3 * xvals**3 + k2 * xvals**2 + k1 * xvals + k0
    liner2 = 1 - np.sum((yvals - liney) ** 2) / np.sum((yvals - np.mean(yvals)) ** 2)
    quadr2 = 1 - np.sum((yvals - quady) ** 2) / np.sum((yvals - np.mean(yvals)) ** 2)
    cuber2 = 1 - np.sum((yvals - cubey) ** 2) / np.sum((yvals - np.mean(yvals)) ** 2)

    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(bounds)
    plt.ylim((0, 1))
    plt.scatter(xvals, yvals, alpha=1)

    plt.plot(xvals, liney)
    plt.plot(xvals, quady)
    plt.plot(xvals, cubey)
    plt.legend([f"line = {np.round(liner2, 5)}", f"quad = {np.round(quadr2, 5)}", f"cube = {np.round(cuber2, 5)}"])

    # plt.plot(xvals, xvals, "k", alpha=0.6)
    # plt.scatter(xvals, (yvals - b) / m) # use this as a reference mask for putting a model on y=x, if possible


# TODO add scatterplots to show relationship between input and output
