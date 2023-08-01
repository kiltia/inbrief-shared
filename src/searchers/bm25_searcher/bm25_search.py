import pathlib
from importlib.machinery import SourceFileLoader

import nltk
from gensim.summarization.bm25 import BM25
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer

utils = SourceFileLoader(
    "utils", str(pathlib.Path(__file__).parent.resolve() / "../../utils.py")
).load_module()

nltk.download("stopwords")
nltk.download("punkt")


def get_tokens(text):
    morph = MorphAnalyzer()
    text = utils.clean_text(text)
    stop_words = stopwords.words("russian")
    stop_words.extend(stopwords.words("english"))
    return [
        morph.normal_forms(i)[0]
        for i in word_tokenize(text.lower())
        if not (i in stop_words)
    ]


class BM25Searcher:
    def __init__(self, texts, embeddings, tokenizer=get_tokens):
        self.corpus = texts
        self.tokenizer = tokenizer
        tok_corpus = [self.tokenizer(text) for text in self.corpus]
        self.bm25 = BM25(tok_corpus)
        self.embeddings = embeddings
        self.used = [False for _ in self.corpus]

    def search_similar_texts(
        self, text_num, threshold=0.11, semantic_threshold=(0.93, 1)
    ):
        text = self.corpus[text_num]
        query = self.tokenizer(text)
        scores = self.bm25.get_scores(query)
        best_docs_num = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        max_score = scores[best_docs_num[0]]
        top = []
        if max_score == 0:
            return []
        for num in best_docs_num:
            score = scores[num] / max_score
            if score > threshold:
                top.append(
                    (
                        utils.get_class_score(
                            self.embeddings[num], self.embeddings[text_num]
                        ),
                        num,
                    )
                )
            else:
                break
        top = sorted(top, reverse=True)
        return [
            pos[1]
            for pos in top
            if semantic_threshold[0] <= pos[0] <= semantic_threshold[1]
        ]

    def search_in_deep(self, text_num, depth=3, min_samples=2, **args):
        texts = []
        searching_texts = [(text_num, 0)]
        while searching_texts:
            cur_text = searching_texts.pop()
            if not (self.used[cur_text[0]]) and cur_text[1] < depth:
                searching_texts.extend(
                    [
                        (i, cur_text[1] + 1)
                        for i in self.search_similar_texts(cur_text[0], **args)
                    ]
                )
                self.used[cur_text[0]] = True
                texts.append(cur_text[0])
        noise = False
        if len(texts) < min_samples:
            noise = True
        return texts, noise
