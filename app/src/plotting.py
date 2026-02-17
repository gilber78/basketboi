import numpy as np
import matplotlib.pyplot as plt


def plot_2d_histogram(x, y, title, binwidth=1, xlabel="Predicted values", ylabel="True values"):
    plt.figure()
    plt.plot(y, y, "w", alpha=0.6)
    plt.hist2d(x, y, bins=[int(max(x) - min(x) / binwidth) + 1, int(max(y) - min(y) / binwidth) + 1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_pdf_function(x, y, title, binwidth=0.05, bounds=(0, 1), xlabel="Predicted Probability", ylabel="True Probability"):
    bins = np.linspace(bounds[0], bounds[1], int(1 / binwidth) + 1)
    xvals = [(bins[i] + bins[i - 1]) / 2 for i in range(1, len(bins))]
    yvals = []
    for i in range(1, len(bins)):
        y_mask = y[(bins[i - 1] <= x) & (x <= bins[i])]
        yvals.append(sum(y_mask) / len(y_mask))

    plt.figure()
    plt.plot(xvals, xvals, "k", alpha=0.6)
    plt.scatter(xvals, yvals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# TODO add scatterplots to show relationship between input and output
