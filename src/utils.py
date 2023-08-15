import re
import string
import numpy as np
from numpy.linalg import norm
from patterns import emoji_pattern


def cos_sim(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


def get_class_score(embedding1, embedding2, scorer=cos_sim):
    return scorer(np.array(embedding1), np.array(embedding2))


def clean_text(text):
    text = re.sub(
        r"https?:\/\/.*[\r\n]*", "", emoji_pattern.sub(r"", text), flags=re.MULTILINE
    )
    text = re.sub(r"@.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"#.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub("[\[(\])]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation + "«»"))
    return text
