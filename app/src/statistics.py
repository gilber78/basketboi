import numpy as np


def calc_ROC_curve(x, y, binwidth=0.01, bounds=(0, 1)):
    class_thresholds = np.linspace(bounds[0], bounds[1], int((bounds[1] - bounds[0]) / binwidth) + 1)[1:-1]
    TPR = []
    FPR = []
    for threshold in class_thresholds:
        predictions = [1 if xval >= threshold else 0 for xval in x]
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i, p in enumerate(predictions):
            match (p - 2) * (y[i] + 2):
                case -4:
                    TN += 1
                case -6:
                    FN += 1
                case -2:
                    FP += 1
                case -3:
                    TP += 1
                case _:
                    pass
        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))
    sort_indeces = np.argsort(FPR)
    TPR_sorted = np.array(TPR)[sort_indeces]
    FPR_sorted = np.array(FPR)[sort_indeces]
    AUC = np.trapz(TPR_sorted, FPR_sorted)

    return TPR_sorted, FPR_sorted, AUC


def calc_brier_score(x, y):
    brier_score = 0
    for i in range(len(x)):
        brier_score += (x[i] - y[i]) ** 2
    brier_score /= len(x)
    return brier_score


def calc_ECE_score(x, y, binwidth=0.05, bounds=(0, 1)):
    bins = np.linspace(bounds[0], bounds[1], int((bounds[1] - bounds[0]) / binwidth) + 1)
    vals = []
    N = len(x)
    for i in range(1, len(bins)):
        y_mask = y[(bins[i - 1] < x) & (x <= bins[i])]
        n_k = len(y_mask)
        p_k = sum(y_mask) / len(y_mask)
        c_k = np.mean([bins[i - 1], bins[i]])
        vals.append((n_k / N) * np.abs(p_k - c_k))
    return sum(vals)


def calc_calibrated_slope_intercept(x, y, binwidth=0.05, bounds=(0, 1)):
    bins = np.linspace(bounds[0], bounds[1], int((bounds[1] - bounds[0]) / binwidth) + 1)
    xvals = [(bins[i] + bins[i - 1]) / 2 for i in range(1, len(bins))]
    yvals = []
    for i in range(1, len(bins)):
        y_mask = y[(bins[i - 1] < x) & (x <= bins[i])]
        yvals.append(sum(y_mask) / len(y_mask))
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    m, b = np.polyfit(xvals, yvals, 1)
    return xvals, yvals, m, b
