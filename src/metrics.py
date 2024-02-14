import numpy as np
from hdbscan import validity_index
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from utils import noises_to_single_cluster, normalize


def silhouette(X, labels, metric="cityblock"):
    labels_copy = noises_to_single_cluster(labels)
    try:
        return silhouette_score(X, labels_copy, metric=metric)
    except ValueError:
        return None


def calinski_harabasz(X, labels, **kwargs):
    labels_copy = noises_to_single_cluster(labels)
    try:
        return calinski_harabasz_score(X, labels_copy)
    except ValueError:
        return None


def weighted_metrics(X, labels, metric):
    labels_copy = noises_to_single_cluster(labels)
    try:
        return calinski_harabasz_score(X, labels_copy), silhouette_score(
            X, labels_copy, metric=metric
        )
    except ValueError:
        return None


def business_metric(X, labels, metric=None):
    validity = (validity_index(X, labels, metric=metric) + 1) / 2
    if np.isnan(validity) or max(labels) == -1:
        return None
    n_noise = len(labels[labels == -1])
    noise_penalty = 1 - n_noise / len(labels)
    normalized_num_clusters = (max(labels) + 1) / len(labels)
    clusters_num_samples = [
        len(labels[labels == i]) for i in range(max(labels) + 1)
    ]
    z_scores = normalize(
        (clusters_num_samples - np.mean(clusters_num_samples))
        / np.std(clusters_num_samples)
    )
    clusters_size_penalty = 2 / (
        1 / np.median(z_scores) + 1 / np.mean(z_scores)
    )
    return 4 / (
        1 / validity
        + 1 / noise_penalty
        + 1 / normalized_num_clusters
        + 1 / clusters_size_penalty
    )
