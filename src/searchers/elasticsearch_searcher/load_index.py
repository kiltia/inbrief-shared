import pandas as pd
from config import ElasticSettings
from elasticsearch import Elasticsearch
from elasticsearch_searcher import add_documents, drop_index

if __name__ == "__main__":
    elastic_config = ElasticSettings()
    es_client = Elasticsearch(
        elastic_config.host,
        ca_certs=elastic_config.ca_certs,
        basic_auth=("elastic", elastic_config.password),
    )
    drop_index(es_client, elastic_config.index_name)
    df = pd.read_json("../../../data/preprocessed.json", convert_dates=True)
    add_documents(df, ["text", "date"], elastic_config.index_name, es_client)
