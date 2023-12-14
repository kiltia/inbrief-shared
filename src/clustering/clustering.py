from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.metrics import silhouette_score


class Clustering:
    def __init__(self, method):
        self.method = getattr(self, "_" + method)

    def _dbscan(self, embeddings, **args):
        clustering = DBSCAN(**args).fit(embeddings)
        labels = clustering.labels_
        clusters = [[] for _ in range(max(labels) + 2)]
        for i in range(len(labels)):
            clusters[labels[i]].append(i)
        return clusters

    def _kmeans(
        self, embeddings, min_samples=2, max_clusters_num=None, **args
    ):
        optimal_cluster_num = self._find_optimal_clusters_num(
            embeddings, KMeans, max_clusters_num, **args
        )
        clustering = KMeans(n_clusters=optimal_cluster_num, **args).fit(
            embeddings
        )
        return self._processing_labels(clustering.labels_, min_samples)

    def _affinity_propagation(self, embeddings, min_samples=2, **args):
        clustering = AffinityPropagation(**args).fit(embeddings)
        return self._processing_labels(clustering.labels_, min_samples)

    def _optics(self, embeddings, **args):
        clustering = OPTICS(**args).fit(embeddings)
        labels = clustering.labels_
        clusters = [[] for _ in range(max(labels) + 2)]
        for i in range(len(labels)):
            clusters[labels[i]].append(i)
        return clusters

    def _spectral_clustering(
        self, embeddings, min_samples=2, max_clusters_num=None, **args
    ):
        optimal_cluster_num = self._find_optimal_clusters_num(
            embeddings, SpectralClustering, max_clusters_num, **args
        )
        clustering = SpectralClustering(
            n_clusters=optimal_cluster_num, **args
        ).fit(embeddings)
        return self._processing_labels(clustering.labels_, min_samples)

    def _agglomerative_clustering(
        self, embeddings, min_samples=2, max_clusters_num=None, **args
    ):
        optimal_cluster_num = self._find_optimal_clusters_num(
            embeddings, AgglomerativeClustering, max_clusters_num, **args
        )
        clustering = AgglomerativeClustering(
            n_clusters=optimal_cluster_num, **args
        ).fit(embeddings)
        return self._processing_labels(clustering.labels_, min_samples)

    def _find_optimal_clusters_num(
        self, embeddings, alg, max_clusters_num=None, **args
    ):
        max_silhouette_score = -1
        optimal_cluster_num = 1
        for k in range(2, len(embeddings)):
            algo = alg(n_clusters=k, **args).fit(embeddings)
            cur_silhouette_score = silhouette_score(embeddings, algo.labels_)
            if cur_silhouette_score > max_silhouette_score:
                max_silhouette_score = cur_silhouette_score
                optimal_cluster_num = k
            if max_clusters_num is not None:
                optimal_cluster_num = min(
                    max_clusters_num, optimal_cluster_num
                )
        return optimal_cluster_num

    def _processing_labels(self, labels, min_samples):
        clusters = [[] for _ in range(max(labels) + 1)]
        for i in range(len(labels)):
            clusters[labels[i]].append(i)
        new_clusters, noise = [], []
        for cluster in clusters:
            if len(cluster) >= min_samples:
                new_clusters.append(cluster)
            else:
                noise.extend(cluster)
        new_clusters.append(noise)
        return new_clusters

    def get_clusters(self, embeddings, **args):
        return self.method(embeddings, **args)
