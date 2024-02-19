import json
import logging

import metrics
import numpy as np
from clustering import get_clustering_method
from sklearn.decomposition import PCA
from utils import normalize

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

    def get_stories(
        self,
        method_name,
        params_range,
        immutable_config,
        *,
        n_components=None,
        return_plot_data=False,
    ):
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

        embeddings = np.array(embeddings)

        method = get_clustering_method(method_name.value)(immutable_config)
        scorer = getattr(metrics, self.scorer)

        if n_components is not None and n_components < len(embeddings):
            embeddings = PCA(n_components, svd_solver="full").fit_transform(
                embeddings
            )

        if self.scorer == LinkingScorer.WEIGHTED_METRICS:
            ranked_entries = method.fine_tune(
                embeddings, scorer, self.metric, params_range, sort=False
            )
            calinski_harabasz = normalize(
                np.array([i[0][0] for i in ranked_entries])
            )
            silhouette = normalize(np.array([i[0][1] for i in ranked_entries]))
            scores = list(zip(calinski_harabasz, silhouette, strict=False))
            for i in range(len(scores)):
                ranked_entries[i] = (
                    2
                    * scores[i][0]
                    * scores[i][1]
                    / (scores[i][0] + scores[i][1]),
                    ranked_entries[i][1],
                )
            ranked_entries = sorted(
                ranked_entries, key=lambda x: x[0], reverse=True
            )
        else:
            ranked_entries = method.fine_tune(
                embeddings, scorer, self.metric, params_range, sort=True
            )

        results = []
        if return_plot_data:
            for rank_entry in ranked_entries:
                stories_nums = method.fit(embeddings, rank_entry[1])
                results.append(
                    {
                        "metadata": {
                            "score": rank_entry[0],
                            "config": rank_entry[1],
                        },
                        "stories_nums": stories_nums,
                    }
                )
        else:
            if not ranked_entries:
                return [
                    {"stories_nums": [[0 for _ in range(len(embeddings))]]}
                ], embeddings

            stories_nums = method.fit(embeddings, ranked_entries[0][1])
            results.append(
                {
                    "metadata": {
                        "score": ranked_entries[0][0],
                        "config": ranked_entries[0][1],
                    },
                    "stories_nums": stories_nums,
                }
            )

        return results, embeddings
