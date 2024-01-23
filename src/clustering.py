import hdbscan
import sklearn.cluster as cls
import numpy as np


def get_clustering_method(method_name: str):
    candidates = BaseCluster.__subclasses__()
    for alg in candidates:
        if alg.get_label() == method_name:
            return alg


class BaseCluster:
    def __init__(self, obj, immutable_config={}):
        self.obj = obj
        self.immutable_config = immutable_config

    def fit(self, X, config, return_labels=False):
        cl = self.obj(**config, **self.immutable_config)
        labels = cl.fit_predict(X)
        clusters = self._form_clusters(labels)
        if return_labels:
            return clusters, labels
        return clusters

    def _form_clusters(self, labels):
        clusters = [[] for _ in range(max(labels) + 2)]
        for i in range(len(labels)):
            clusters[labels[i]].append(i)
        return clusters

    def link_text(clusters, texts):
        return list(map(lambda x: list(map(lambda y: texts[y], x)), clusters))

    @classmethod
    def get_label(self):
        return self.__name__.lower()


class OPTICS(BaseCluster):
    def __init__(self, immutable_config):
        super().__init__(cls.OPTICS, immutable_config)

    def fine_tune(self, X, scorer, metric, params_range, sort=False):
        results = []
        eps_range = params_range["eps"]
        min_samples_range = params_range["min_samples"]
        eps = eps_range[0]
        while eps < eps_range[1]:
            for i in range(
                min_samples_range[0], min(min_samples_range[1], len(X))
            ):
                labels = self.obj(
                    min_samples=i, max_eps=eps, **self.immutable_config
                ).fit_predict(X)
                if len(np.unique(labels)) == 1:
                    eps += 0.05
                    continue
                metric_value = scorer(X, labels, metric=metric)
                if not np.isnan(metric_value):
                    results.append(
                        (metric_value, {"min_samples": i, "max_eps": eps})
                    )
            eps += 0.05

        if sort:
            return sorted(results, key=lambda x: x[0], reverse=True)
        return results


class HDBSCAN(BaseCluster):
    def __init__(self, immutable_config):
        super().__init__(hdbscan.HDBSCAN, immutable_config)

    def fine_tune(self, X, scorer, metric, params_range, sort=False):
        results = []
        min_cluster_size_range = params_range["min_cluster_size"]
        for i in range(
            min_cluster_size_range[0], min(min_cluster_size_range[1], len(X))
        ):
            labels = self.obj(min_cluster_size=i).fit_predict(X)
            if len(np.unique(labels)) == 1:
                continue
            metric_value = scorer(X, labels, metric=metric)
            if not np.isnan(metric_value):
                results.append((metric_value, {"min_cluster_size": i}))

        if sort:
            return sorted(results, key=lambda x: x[0], reverse=True)
        return results


class KMeans(BaseCluster):
    def __init__(self, immutable_config):
        super().__init__(cls.KMeans, immutable_config)

    def fine_tune(self, X, scorer, metric, params_range, sort=False):
        results = []
        n_clusters_range = params_range["n_clusters"]
        for n_clusters in range(
            n_clusters_range[0], min(n_clusters_range[1], len(X))
        ):
            labels = self.obj(
                n_clusters=n_clusters, **self.immutable_config
            ).fit_predict(X)
            metric_value = scorer(X, labels, metric=metric)
            if not np.isnan(metric_value):
                results.append((metric_value, {"n_clusters": n_clusters}))

        if sort:
            return sorted(results, key=lambda x: x[0], reverse=True)
        return results


class AGGLOMERATIVE(BaseCluster):
    def __init__(self, immutable_config):
        super().__init__(cls.AgglomerativeClustering)

    def fine_tune(self, X, scorer, metric, params_range, sort=False):
        results = []
        distance_threshold_range = params_range["distance_threshold"]
        distance_threshold = distance_threshold_range[0]
        while distance_threshold < distance_threshold_range[1]:
            labels = self.obj(
                distance_threshold=distance_threshold,
                n_clusters=None,
                **self.immutable_config,
            ).fit_predict(X)
            if len(np.unique(labels)) == 1:
                break
            metric_value = scorer(X, labels, metric=metric)
            if not np.isnan(metric_value):
                results.append(
                    (metric_value, {"distance_threshold": distance_threshold})
                )
            distance_threshold += 0.05

        if sort:
            return sorted(results, key=lambda x: x[0], reverse=True)
        return results
