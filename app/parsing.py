#!/usr/bin/python

import os
import sys
import pandas as pd 
import numpy as np 
from glob import glob
import argparse
import json
import gc



def load_judgement_files(folder):

    ## Load and return all judgment files in specified folder

    fs = glob(os.path.join(folder, '*_Technical.txt'))

    assert len(fs) != 0, 'Got an empty folder!'

    return fs


def read_files(file_path):
    ## Read judgement file and corresponding metadata  and return content
    

    with open(file_path, 'r', encoding ='utf-8') as f:
        data = [line.strip() for line in f.readlines()]
        data = [ sent for sent in data if sent not in ['__paragraph__', '__section__', 'JUDGEMENT']]

    judgement = pd.DataFrame({'PARAGRAPH' : data})

    with open(file_path.replace('Technical.txt', 'Metadata.json'), 'r',  encoding ='utf-8') as f:
        meta = json.load(f)
    
    return judgement, meta


def paragraph_splitter(paragraphs : str = '', chunks : int = 5):

    " split paragraphs into non-overlapping chunks of paragraphs"

    paragraph_chunks = []

    for i in range(0, len(paragraphs), chunks):

        txt_chunk = paragraphs[i : (i + chunks)]

        paragraph_chunks.append(txt_chunk)

    return paragraph_chunks


def main(cfg):

    fps = load_judgement_files(cfg.input_folder)

    data = [ read_files(fp) for fp in fps]

    dfs = [o[0] for o in data ]
    mfs = [o[1] for o in data ]


    paragraphs = [df['PARAGRAPH'].values.tolist() for df in dfs]

    passages = ['\n'.join(pp) for pp in paragraphs]

    passages = [paragraph_splitter(passage, chunks = cfg.chunk_splits) for passage in passages]

    
    passage_metadata_pair = pd.DataFrame()
    passage_metadata_pair['Passage'] = passages
    passage_metadata_pair['Metadata'] = mfs

    passage_metadata_pair.to_csv(f'{cfg.output_folder}/passage_metadata.csv', index=False)

    _=gc.collect()


if  __name__ == "__main__":

    parser = argparse.ArgumentParser("Document Parser")
    parser.add_argument('--input_folder', type=str, default='.\Corpus')
    parser.add_argument('--output_folder', type= str, default='.\docs')
    parser.add_argument('--chunk_splits', type = int, default = 5)

    args= parser.parse_args()

    main(args)




    
