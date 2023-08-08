import pandas as pd
import pickle
from tqdm import tqdm
import random
import re
import numpy as np


def pick_tag_sent(thres): 
    df = pd.read_csv(f'/Data/jiwon/7-classes/230329-ner-dataset-full({thres})-o-tag.csv', low_memory = False)
    sents = df['sent'].to_list()
    tags = df['tag'].to_list()

    yes_list = []
    no_list = []

    for sent, tag in zip(sents, tags):
        if 'DS-B' in tag or 'METH-B' in tag:
            yes_list.append([sent, tag])
        else: no_list.append([sent, tag])

    random.shuffle(no_list)
    final_list = yes_list

    print(len(final_list))
    df = pd.DataFrame(final_list, columns=['sent', 'tag'])
    df.to_csv('230329-dataset-for-NERtraining-full.csv', encoding='utf8', index=False)


def pick_O_sents(): 
    df = pd.read_csv(f'/Data/jiwon/7-classes/230329-ner-dataset-full(0.1)-o-tag.csv', low_memory = False)
    sents = df['sent'].to_list()
    tags = df['tag'].to_list()

    yes_list = []
    no_list = []

    for sent, tag in zip(sents, tags):
        if 'DS-B' in tag or 'METH-B' in tag:
            yes_list.append([sent, tag])
        else: no_list.append([sent, tag])

    random.shuffle(no_list)
    final_list = no_list[:30000]

    print(len(final_list))
    df = pd.DataFrame(final_list, columns=['sent', 'tag'])
    df.to_csv('230403-dataset-for-NERtraining-O-tag-full.csv', encoding='utf8', index=False)


if __name__ == '__main__':
    #pick_tag_sent(0.1)
    #pick_O_sents()
    
