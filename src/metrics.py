import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from utils import processing_noise


def silhouette(X, labels, metric="cityblock"):
    labels_copy = processing_noise(labels)
    try:
        return silhouette_score(X, labels_copy, metric=metric)
    except ValueError:
        return np.nan


def calinski_harabasz(X, labels, **kwargs):
    labels_copy = processing_noise(labels)
    try:
        return calinski_harabasz_score(X, labels_copy)
    except ValueError:
        return np.nan


def weighted_metrics(X, labels, metric):
    labels_copy = processing_noise(labels)
    try:
        return 0.01 * calinski_harabasz_score(
            X, labels_copy
        ) + silhouette_score(X, labels_copy, metric=metric)
    except ValueError:
        return np.nan
