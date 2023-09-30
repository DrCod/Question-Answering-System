import os
import sys
import pandas as pd 
import argparse
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk


def create_index(client):

    """ Creates an elastic search index"""

    client.indices.create(
        index='question-anwering-system',
        body = {
            "settings" : {'number_of_shards' : 1},
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

def generate_actions(cfg):

    with open(cfg.data_dir, mode="r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            doc = {
                
                "passage" : row['Passage'],
                "metadata" : row['Metadata'],
                "embeddings" : row['Embedding']

            }

            yield doc

def main(args):

    client = Elasticsearch(args.host)

    print("Creating an index...")
    create_index(client)

    print("Indexing documents...")
    successes = 0
    for ok, action in streaming_bulk(
        client=client, index="question-anwering-system", actions=generate_actions(args),
    ):
        successes += ok


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default = '\docs\passage_metadata.csv')
    parser.add_argument('--host', type = str, default = 'http://localhost:9200')
    parser.add_argument('--elastic_password', type= str, default ='elastic')

    args = parser.parse_args()

    main(args)