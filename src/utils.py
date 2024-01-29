import numpy as np
from copy import deepcopy
from numpy.linalg import norm


def cos_sim(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (
        norm(embedding1) * norm(embedding2)
    )


def get_class_score(embedding1, embedding2, scorer=cos_sim):
    return scorer(np.array(embedding1), np.array(embedding2))

def processing_noise(labels):
    labels_copy = deepcopy(labels)
    cur = np.max(labels) + 1
    for i in range(len(labels_copy)):
        if labels_copy[i] == -1:
            labels_copy[i] = cur
            cur += 1
    return labels_copy
