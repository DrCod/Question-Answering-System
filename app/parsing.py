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
          
        data = [line.strip() for line in f.readlines() if line != '']

        data = data[2:]

    judgement = ''

    for line in data:

        if line not in ['__paragraph__', ''] :
            judgement += str(line) + '\n'

    with open(file_path.replace('Technical.txt', 'Metadata.json'), 'r',  encoding ='utf-8') as f:
        meta = json.load(f)
    
    return judgement, meta, len(data)


def chunk_splitter(paragraphs : str = '', chunks : int = 5):

    "split paragraphs into non-overlapping chunks of paragraphs"


    sents = paragraphs.split('\n')

    paragraph_chunks = []

    for i in range(0, len(sents), chunks):

        txt_chunk = sents[i : (i + chunks)]

        paragraph_chunks.append('\n'.join(txt_chunk))

    print(f'Chunks created : {len(paragraph_chunks)}')
    print()
    print(f'Chunk example : {paragraph_chunks[0]}')

    return paragraph_chunks

def main(cfg):

    fps = load_judgement_files(cfg.input_folder)

    data = [ read_files(fp) for fp in fps]

    # unpack data
    paragraphs = [o[0] for o in data]
    mfs = [o[1] for o in data]

    passages = [chunk_splitter(paragraphs=paragraph, chunks = cfg.chunk_splits) for paragraph in paragraphs]
    
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




    
