import json

from clustering.clustering import Clustering

from shared.models import ClusteringMethod, EmbeddingSource


class Matcher:
    def __init__(self, entities, embedding_source: EmbeddingSource):
        self.entities = entities
        self.embedding_source = embedding_source

    def get_stories(self, method, **args):
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
        if ClusteringMethod.has_value(method.value):
            stories_nums = Clustering(method.value).get_clusters(
                embeddings, **args
            )

        stories = []
        for i in range(len(stories_nums)):
            story = []
            for num in stories_nums[i]:
                story.append(self.entities[num])
            stories.append(story)
        return stories, stories_nums
