import uuid
from datetime import datetime
from importlib.machinery import SourceFileLoader

from tqdm import tqdm

utils = SourceFileLoader("utils", "../utils.py").load_module()


def add_documents(df, columns, index_name, client):
    for i in tqdm(range(df.shape[0])):
        doc = {}
        values = df.loc[i][columns].values
        for j in range(len(columns)):
            doc[columns[j]] = values[j]
        client.index(index=index_name, id=str(uuid.uuid4()), document=doc)
    client.indices.refresh(index=index_name)


def search_similar_docs(
    text,
    client,
    index_name,
    start_date="2001-01-01T00:00:00",
    end_date=None,
    k=50,
    bm25_threshold=0.155,
    semantic_threshold=(0.93, 0.9946),
):
    text = utils.clean_text(text)
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    query = {
        "size": k,
        "query": {
            "bool": {
                "should": [{"match": {"text": text}}],
                "filter": [
                    {
                        "range": {
                            "date": {
                                "gte": start_date,
                                "lte": end_date,
                            }
                        }
                    }
                ],
            }
        },
    }
    result = client.search(index=index_name, body=query)
    max_score = result["hits"]["max_score"]
    top = []
    for hit in result["hits"]["hits"]:
        score = hit["_score"] / max_score
        if score > bm25_threshold:
            top.append(
                (
                    utils.get_class_score(hit["_source"]["text"], text),
                    hit["_source"]["text"],
                    hit["_source"]["date"],
                )
            )
        else:
            break
    top = sorted(top, reverse=True)
    return [
        (pos[1], pos[2])
        for pos in top
        if semantic_threshold[0] <= pos[0] <= semantic_threshold[1]
    ]


def search_in_depth(text, depth=1, **args):
    pass


def drop_index(client, index_name):
    client.options(ignore_status=[400, 404]).indices.delete(index=index_name)
