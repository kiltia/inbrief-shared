import logging
import os
import shutil
from typing import List

import fasttext as ft
import fasttext.util
import openai_api
from rb_tocase import Case
from transformers import AutoModel, AutoTokenizer

from shared.utils import CACHE_PATH, clean_text

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    def get_embeddings(self, inputs: list, **kwargs):
        pass

    @classmethod
    def get_label(self):
        return Case.to_kebab(self.__name__)


class OpenAiEmbedder(EmbeddingProvider):
    # TODO(nrydanov): Pass default parameter from configuration file instead
    def __init__(self, model: str = "text-embedding-ada-002") -> None:
        self.model = model

    def get_embeddings(self, inputs: list, **kwargs):
        return openai_api.get_embeddings(inputs, self.model)


class MiniLmEmbedder(EmbeddingProvider):
    TOKENIZER = "nreimers/mmarco-mMiniLMv2-L12-H384-v1"

    # TODO(nrydanov): Pass default parameter from configuration file instead
    def __init__(self, weights_path=None) -> None:
        if not weights_path:
            weights_path = f"{CACHE_PATH}/mini-lm"
        self.tokenizer = AutoTokenizer.from_pretrained(
            MiniLmEmbedder.TOKENIZER, cache_dir=weights_path
        )
        self.model = AutoModel.from_pretrained(weights_path)

    def get_embeddings(self, inputs: list, **kwargs):
        results = []
        for input in inputs:
            embedding = (
                self.model(
                    **self.tokenizer(
                        input,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                )["pooler_output"]
                .cpu()
                .detach()
                .numpy()[0]
                .tolist()
            )

            results.append(embedding)
        return results


class FastTextEmbedder(EmbeddingProvider):
    # TODO(nrydanov): Pass default parameter from configuration file instead
    def __init__(self, weights_path=None):
        if not weights_path:
            weights_path = f"{CACHE_PATH}/cc.ru.300.bin"

        if not os.path.exists(weights_path):
            ft.util.download_model("ru")
            shutil.move("cc.ru.300.bin", weights_path)

        self.model = ft.load_model(weights_path)

    def get_embeddings(self, inputs):
        embeddings = []

        for input in inputs:
            text = clean_text(input)
            embedding = self.model.get_sentence_vector(text.replace("\n", " "))
            embeddings.append(embedding.tolist())

        return embeddings


embedders: List[EmbeddingProvider] = []


def init_embedders():
    global embedders

    candidates = EmbeddingProvider.__subclasses__()
    for embedder in candidates:
        logger.info(f"Started loading {embedder.get_label()}")
        try:
            obj = embedder()
        except Exception as e:
            logging.error(
                f"Got {type(e).__name__} exception while initializing {embedder.get_label()}: {e}"
            )
        embedders.append(obj)
        logger.info(f"Finished loading {embedder.get_label()}")


def get_embedders(names: str):
    global embedders

    if names is None:
        return embedders

    required = list(filter(lambda entry: entry.get_label() in names, embedders))
    return required
