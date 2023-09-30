import os
import pandas as pd
import numpy as np 
from sentence_transformers import SentenceTransformer
import argparse


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



def main(args):

    cfg = ModelConfig

    embedder = get_embedder(cfg)

    qvector = embedder.encode([cfg.query], normalize_embeddings= (args.normalize_embeddings != None ))

    script_query = {

        "script_score" : {
            "query" : {"match_all" : {}},
            "script" : {
                "source": f"cosineSimilarity(params.query_vector, 'embeddings') + 1.0",

                "params" : {"query_vector" : qvector}
            }
        }
    }

    res = elastic.search(args.index_name, script_query, args.n_returns, list_fields_to_return=["passage", "metadata"])

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

    del res_df, res, embedder, qvector; _ = gc.collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--query", type=str, default="sample text" )
    parser.add_argument("--index_name", type=str, default="question-anwering-system")
    parser.add_argument("--n_returns", type=int, default=3)
    parser.add_argument("--normalize_embeddings", action="store_true", default=None)
    parser.add_argument("--output_folder", type=str, default=".\docs")

    args = parser.parse_args()
    main(args)