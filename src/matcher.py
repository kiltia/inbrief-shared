import json
import metrics

from clustering import get_clustering_method
from shared.models import ClusteringMethod, EmbeddingSource

from sklearn.decomposition import PCA


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
        immutable_config={},
        n_components=None,
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

        if n_components is not None:
            embeddings = PCA(n_components).fit_transform(embeddings)

        method = get_clustering_method(method_name.value)(immutable_config)
        best_config = method.fine_tune(
            embeddings, self.scorer, self.metric, params_range, sort=True
        )[0]
        stories_nums = method.fit(embeddings, best_config)

        return stories_nums
