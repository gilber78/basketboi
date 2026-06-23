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
