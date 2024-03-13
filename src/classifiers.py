import logging
from typing import List

import torch
from rb_tocase import Case
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import CLASSES_NAMES

from shared.utils import CACHE_PATH

logger = logging.getLogger("scraper")


class Classifier:
    CLASSES_MAPPING = {i: name for i, name in enumerate(CLASSES_NAMES)}

    def get_labels(self, inputs: List[str], **kwargs):
        pass

    @classmethod
    def get_name(self):
        return Case.to_kebab(self.__name__)


class MiniLmClassifier(Classifier):
    TOKENIZER = "nreimers/mmarco-mMiniLMv2-L12-H384-v1"

    # TODO(sokunkov): Pass default parameter from configuration file instead
    def __init__(self, weights_path=None) -> None:
        if not weights_path:
            weights_path = f"{CACHE_PATH}/mini-lm"
        self.tokenizer = AutoTokenizer.from_pretrained(
            MiniLmClassifier.TOKENIZER, cache_dir=weights_path
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            weights_path
        )

    def get_labels(self, inputs: List[str], **kwargs):
        labels = []
        for input in inputs:
            label = torch.argmax(
                self.model(
                    **self.tokenizer(
                        input,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                ).logits[0]
            ).item()

            labels.append(Classifier.CLASSES_MAPPING[label])
        return labels


def get_classifier(classifier_name: str):
    classifier = None
    candidates = {
        candidate.get_name(): candidate
        for candidate in Classifier.__subclasses__()
    }
    if classifier_name in candidates:
        logger.info(f"Started loading {classifier_name}")
        try:
            classifier = candidates[classifier_name]()
        except Exception as e:
            logging.error(
                f"Got {type(e).__name__} exception while initializing {classifier_name}: {e}"
            )
    elif classifier_name is None:
        logger.warn(
            "Classifier is not defined in config! Classification of scraped texts will not be carried out."
        )
    else:
        logging.error(f"Classifier {classifier_name} doesn't exist")
    return classifier
