from sklearn.cluster import DBSCAN


def get_clusters(embeddings, **args):
    clustering = DBSCAN(**args).fit(embeddings)
    labels = clustering.labels_
    clusters = [[] for _ in range(max(labels) + 2)]
    for i in range(len(labels)):
        clusters[labels[i]].append(i)
    return clusters
