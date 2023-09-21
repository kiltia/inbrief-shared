## Linker

This module is used to combine texts in stories.

### About this service
This module contains two methods of matching texts: clustering and hand-wrote search engine.

#### DBScan (module clustering.clustering)
This method based on algorithm DBSCAN. We chose this algorithm because it work fast and doesn't require specifying number of clusters. This is good in our project, where matcher doesn't know how many stories it can match from gotten texts.

This method of matching use next arguments:

* `eps`: float, default=0.425 \
  The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
* `min_samples`: int, default=2 \
  The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
* `metric`: str, default='l2' \
  The metric to use, when calculating distance between embeddings of texts. \
  Available metrics: cityblock, cosine, euclidean, l1, l2, manhattan, braycurtis, canberra, chebyshev, correlation, dice, hamming, jaccard, mahalanobis, minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, yule.

#### BM25 searcher (module searchers.bm25_searcher.bm25search)
This method is a hand-wrote searcher based on algorithm BM25 for searching similar texts. We have modified this by adding the following changes:
1. We added second level reranking and filtration model based on semantic embeddings. This helped to discard news that were similar in words but different in meaning.
2. We decided to represent the received news as a graph and search for similar news in depth using the dfs algorithm and mark "searched" news. At the same time, limiting this search to a given depth. This allowed us to receive not just similar news, but also full-fledged stories from these news.

This method of matching use next arguments:

* `depth`: int, default=3 \
  The max depth of search in dfs.
* `threshold`: float, default=0.11 \
  The lower bound for BM25 similarity.
* `semantic_threshold`: tuple, default=(0.93, 1.0) \
  The tuple with two float number, which contains upper bound and lower bound acceptable semantic similarity.

### Running

This application is built using FastAPI, so you may use
```
uvicorn --app-dir=src main:app
```
command.

NOTE: running this outside of container requires changing corresponding host
and port in Supervisor service configuration.

But since this service is a part of inbrief project, you may use `docker-compose up`/`docker-compose start linker`
in any child directory. 

## API

Port: 6002 -> 8000

### POST /get_stories
- `texts`: array \
  The array with initial texts.
- `embeddings`: array \
  The array with embeddings of initial texts.
- `method`: enum, default=`"dbscan"` \
  The name of matching method. Available methods: `"dbscan"`, `"bm25"`.
- `config`: dict, optional\
  The dict with parameters for using matching method.

This service returns json with array of stories and array of number initial texts in stories.

Example request:
```
{
  "story" : ["text", "text"],
  "embeddings" : [[0, 1, 2, 0], [2, 1, 0, 3]]
  "method" : "dbscan",
  "config" : {
    "eps" : 0.5,
    "min_samples" : 2,
    "metric" : "l2"
  }
}
```
