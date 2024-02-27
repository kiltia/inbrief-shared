import json
import logging

import metrics
import numpy as np
from clustering import get_clustering_method
from metrics import apply_weighted_scorer
from sklearn.decomposition import PCA

from shared.models import EmbeddingSource, LinkingScorer

logger = logging.getLogger("app")


class Matcher:
    def __init__(
        self, entities, embedding_source: EmbeddingSource, scorer, metric
    ):
        self.entities = entities
        self.embedding_source = embedding_source
        self.scorer = scorer
        self.metric = metric

    def _retrieve_embeddings(self):
        embs = list(map(lambda x: json.loads(x.embeddings), self.entities))
        embs = {k: [entity[k] for entity in embs] for k in embs[0]}
        match self.embedding_source:
            case EmbeddingSource.FTMLM:
                embeddings = [
                    x + y
                    for x, y in zip(
                        embs["mini-lm-embedder"],
                        embs["fast-text-embedder"],
                        strict=True,
                    )
                ]
            case EmbeddingSource.OPENAI:
                embeddings = embs["open-ai-embedder"]
            case EmbeddingSource.MLM:
                embeddings = embs["mini-lm-embedder"]

        return np.array(embeddings)

    def _form_entry(method, embeddings, entry):
        stories_nums = method.fit(embeddings, entry[1])
        return {
            "metadata": {
                "score": entry[0],
                "config": entry[1],
            },
            "stories_nums": stories_nums,
        }

    def get_stories(
        self,
        method_name,
        params_range,
        immutable_config,
        *,
        n_components=None,
        return_plot_data=False,
    ):
        embeddings = self._retrieve_embeddings()
        method = get_clustering_method(method_name.value)(immutable_config)
        scorer = getattr(metrics, self.scorer)

        if n_components is not None and n_components < len(embeddings):
            embeddings = PCA(n_components, svd_solver="full").fit_transform(
                embeddings
            )

        if self.scorer == LinkingScorer.WEIGHTED_SCORER:
            ranked_entries = apply_weighted_scorer(
                embeddings, method, scorer, params_range, self.metric
            )
        else:
            ranked_entries = method.fine_tune(
                embeddings, scorer, self.metric, params_range, sort=True
            )

        results = []
        logger.debug(ranked_entries)
        if return_plot_data:
            for rank_entry in ranked_entries:
                results.append(
                    Matcher._form_entry(method, embeddings, rank_entry)
                )
        else:
            if not ranked_entries:
                return [
                    {"stories_nums": [[i for i in range(len(embeddings))], []]}
                ], embeddings

            results.append(
                Matcher._form_entry(method, embeddings, ranked_entries[0])
            )

        return results, embeddings
