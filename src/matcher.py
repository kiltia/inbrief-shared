import json

import metrics
import numpy as np
from clustering import get_clustering_method
from sklearn.decomposition import PCA

from shared.models import EmbeddingSource


class Matcher:
    def __init__(
        self, entities, embedding_source: EmbeddingSource, scorer, metric
    ):
        self.entities = entities
        self.embedding_source = embedding_source
        self.scorer = getattr(metrics, scorer)
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

        if n_components is not None and n_components < len(embeddings):
            embeddings = PCA(n_components).fit_transform(embeddings)

        method = get_clustering_method(method_name.value)(immutable_config)
        ranked_entries = method.fine_tune(
            embeddings, self.scorer, self.metric, params_range, sort=True
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
            stories_nums = method.fit(embeddings, ranked_entries[0][1])
            results.append(
                {
                    "metadata": {
                        "value": ranked_entries[0][0],
                        "config": ranked_entries[0][1],
                    },
                    "stories_nums": stories_nums,
                }
            )

        return results, embeddings
