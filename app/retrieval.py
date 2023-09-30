import os
import pandas as pd
import numpy as np 
from elasticsearch import Elasticsearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer
from elasticsearch.exceptions import ConnectionError, NotFoundError
import argparse

import gc

class ModelConfig:

    MODEL_BACKBONE = 'MarcoMancini/low-law-emb'
    MAX_SEQ_LEN = 64


def get_embedder(cfg):

    """
    Initialize mdoel
    """

    model = SentenceTransformer(cfg.MODEL_BACKBONE)
    model.max_seq_length = cfg.MAX_SEQ_LEN 

    return model


def search(es, args, query_script):

    try:

        res = es.search(index=args.index_name, 
                        body= { 
                            "query": query_script, 
                            "size": args.n_returns, 
                            "_source": ["passage", "metadata"]
                            },)

        return res

    except ConnectionError:
        print('Could not connect to elasticsearch server!')
    except NotFoundError:
        print(f"Index qna could not be found!")


def main(args):

    cfg = ModelConfig

    embedder = get_embedder(cfg)

    qvector = embedder.encode([args.query], normalize_embeddings= (args.normalize_embeddings is not None ))

    es = Elasticsearch(args.host, verify_certs=False, ca_certs=False,
     http_auth=('elastic', args.password), timeout=30, max_retries=10, retry_on_timeout=True)


    script_query = {

        "script_score" : {
            "query" : {"match_all" : {}},
            "script" : {
                "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",

                "params" : {"query_vector" : qvector}
            }
        }
    }

    res = search(es, args,script_query)

    del embedder, qvector; _=gc.collect()

    if res:

        results = []

        for hit in res["hits"]["hits"]:
            results.append([hit["_source"]["passage"], hit["_source"]["metadata"], hit["_score"] - 1 ])


        res_df = pd.DataFrame()
        res_df["Question"] = args.query
        res_df["Passage 1"] = results[0][0]
        res_df["Relevance Score 1"] = results[0][2]
        res_df["Passage Metadata 1"] = results[0][1]
        res_df["Passage 2"] = results[0][0]
        res_df["Relevance Score 2"] = results[0][2]
        res_df["Passage Metadata 2"] = results[0][1]
        res_df["Passage 3"] = results[0][0]
        res_df["Relevance Score 3"] = results[0][2]
        res_df["Passage Metadata 3"] = results[0][1]

        res_df.to_csv(f"{args.output_folder}\question_answers.csv", index=False)

        del res_df; _ = gc.collect();


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--query", type=str, default="sample text" )
    parser.add_argument("--index_name", type=str, default="qna")
    parser.add_argument("--n_returns", type=int, default=3)
    parser.add_argument("--normalize_embeddings", action="store_true", default=None)
    parser.add_argument("--output_folder", type=str, default=".\docs")
    parser.add_argument("--password", type=str, default="pass")
    parser.add_argument("--host", type=str, default="https://...")

    args = parser.parse_args()
    main(args)