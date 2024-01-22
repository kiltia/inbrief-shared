import numpy as np
from copy import deepcopy
from sklearn.metrics import silhouette_score


def silhouette(X, labels, metric="cityblock"):
    labels_copy = deepcopy(labels)
    cur = np.max(labels) + 1
    for i in range(len(labels_copy)):
        if labels_copy[i] == -1:
            labels_copy[i] = cur
            cur += 1
    return silhouette_score(X, labels_copy, metric=metric)
