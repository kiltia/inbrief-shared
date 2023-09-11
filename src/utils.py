import numpy as np
from numpy.linalg import norm


def cos_sim(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


def get_class_score(embedding1, embedding2, scorer=cos_sim):
    return scorer(np.array(embedding1), np.array(embedding2))
