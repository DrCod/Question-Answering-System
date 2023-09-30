import os
import sys
import pandas as pd 
import argparse
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

import csv

def create_index(client):

    """ Creates an elastic search index"""

    try:
        
        client.indices.create(

            index='qna',
            body = {
                "settings" : {
                    'number_of_shards' : 1,
                    'number_of_replicas' :1
                },
                "mappings" : {

                    "properties" : {

                        "passage" : {"type" : "object"},
                        "metadata" : {"type" : "object"},
                        "embeddings" : {"type" : "dense_vector"},
                } 
                },

            },
            ignore = 400,
        )
    except Exception as e:
        print(e)
        

def generate_actions(cfg):

    df = pd.read_csv(cfg.data_dir)

    for _, row in df.iterrows():

        doc = {
            
            "passage" : row['Passage'],
            "metadata" : row['Metadata'],
            "embeddings" : row['Embedding']

        }

        yield doc

def main(args):

    client = Elasticsearch('https://localhost:9200', verify_certs=False, ca_certs=False, 
    http_auth=('elastic', 'WvODJyTTXsJRVMtfQZgc'), timeout=30, max_retries=10, retry_on_timeout=True)

    print("Creating an index...")
    create_index(client)

    print("Indexing documents...")
    successes = 0
    for ok, action in streaming_bulk(
        client=client, index="qna", actions=generate_actions(args),
    ):
        successes += ok


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default = '\docs\passage_metadata.csv')
    args = parser.parse_args()

    main(args)