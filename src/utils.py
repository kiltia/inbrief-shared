import re

import numpy as np
from patterns import emoji_pattern
from sklearn.metrics.pairwise import cosine_similarity


def get_class_score(embedding1, embedding2, scorer=cosine_similarity):
    return scorer(
        np.array(embedding1).reshape(1, -1), np.array(embedding2).reshape(1, -1)
    )


def clean_text(text):
    text = re.sub(
        r"https?:\/\/.*[\r\n]*", "", emoji_pattern.sub(r"", text), flags=re.MULTILINE
    )
    text = re.sub(r"@.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"#.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub("[\[(\])]", "", text)
    return text
