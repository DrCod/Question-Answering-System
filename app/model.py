import sklearn

from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import torch
import argparse
import pandas as pd
import ast
import gc

class Config:

    MODEL_BACKBONE = 'MarcoMancini/low-law-emb'
    MAX_SEQ_LEN = 64

def model(cfg):

    """
    Initialize mdoel
    """

    model = SentenceTransformer(cfg.MODEL_BACKBONE)
    model.max_seq_length = cfg.MAX_SEQ_LEN 

    return model

def extract_embeddings_df(cfg, args, model):
    """
    Extract passage embeddings from the dataframe
    """
    df = pd.read_csv(args.input_file)

    f = lambda x : ast.literal_eval(x['Passage'])

    df["Passage"] = df.apply(f, axis=1).values

    df['Embedding'] =  df['Passage'].apply(lambda inp : model.encode(inp, normalize_embeddings= (args.normalize_embeddings != None ) ))

    _ = gc.collect()

    return df


def main(args):

    model_cfg = Config

    mm = model(model_cfg)

    emb_df = extract_embeddings_df(model_cfg, args, mm)


    emb_df.to_csv(f'{args.output_folder}\passage_metadata_emb.csv', index=False)

    del emb_df , mm; _= gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='.\docs\passage_metadata.csv')
    parser.add_argument('--output_folder', type= str, default='.\docs')
    parser.add_argument('--normalize_embeddings', action='store_true', default = None)

    args= parser.parse_args()

    main(args)