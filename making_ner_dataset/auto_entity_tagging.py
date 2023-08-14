import pandas as pd
import pickle
from nltk import sent_tokenize
from tqdm import tqdm
import random
import spacy 
import re
import numpy as np


def ngrams(word, n):
    ngram_list = []
    for i in range(0,len(word)-n+1):
        ngram_list.append(word[i:i+n])
    return ngram_list
  

def make_dataset(thres):
    with open('/Data/jiwon/data/dataset_list_variants', 'rb') as f:
        dataset_list = pickle.load(f)
    dataset_list = list(set(dataset_list))
    dataset_list.sort(key=len, reverse=True)
    
    with open('/Data/jiwon/data/method_list_variants', 'rb') as f:
        method_list = pickle.load(f)
    method_list = list(set(method_list))
    method_list.sort(key=len, reverse=True)


    nlp = spacy.load("en_core_web_md")


    df = pd.read_csv('', low_memory = False)
    df_abst = df['full_text'].to_list() + df['abstract'].to_list()
    #df_abst = df['abstract'].to_list()


    final_ner_list = []

    dataset_appear = []
    method_appear = []

    for abst in tqdm(df_abst):
        try: 
            sent_list = nlp(abst)

        except:
            print('!')
            continue

        for sent in sent_list.sents:
            sent = sent.text
            sent = sent.replace('\n', '')
            sent = re.sub(' +', ' ', sent)
            sent = sent.strip()
            sent = repr(sent)
            sent = sent[1:-1]
            tag_sent = sent
            sent_list = sent.split(' ')
            sent_len = len(sent_list)

            for dataset in dataset_list:
                if dataset != '':
                    dataset_len = len(dataset.split(' '))
                    if len(dataset) < 3:
                        for word in sent_list:
                            word_to_compare = word.strip(')''('',''.''['']''{''}''?''!'':'';'' ')
                            if dataset == word_to_compare:
                                if dataset_len == 1:
                                    if word_to_compare.strip()[0].isupper() or word_to_compare.strip()[0].isdigit():
                                        dataset_appear.append(dataset)
                                        tag_sent = tag_sent.replace(word, 'DS-B')
                                elif dataset_len > 1:
                                    if word_to_compare.strip()[0].isupper() or word_to_compare.strip()[0].isdigit():
                                        dataset_appear.append(dataset)
                                        ds_I_list = ['DS-I'] * (dataset_len - 1)
                                        tag_sent = tag_sent.replace(word, 'DS-B ' + ' '.join(ds_I_list))
                    else:
                        if sent_len >= dataset_len:
                            for i in range(0, sent_len - dataset_len + 1):
                                candidate_word = ' '.join(sent_list[i:i+dataset_len])
                                if dataset.lower() in candidate_word.lower():
                                    candidate_word_to_compare = candidate_word.strip(')''('',''.''['']''{''}''?''!'':'';'' ')
                                    a = ngrams(candidate_word_to_compare.strip(), 3)
                                    b = ngrams(dataset, 3)
                                    c = set(a) & set(b)
                                    dice_sim = (2 * len(c)) / (len(a) + len(b))
                                    if dice_sim >= thres:
                                        if dataset_len == 1:
                                            if candidate_word_to_compare.strip()[0].isupper() or candidate_word_to_compare.strip()[0].isdigit():
                                                dataset_appear.append(dataset)
                                                tag_sent = tag_sent.replace(candidate_word, 'DS-B')
                                        elif dataset_len > 1:
                                            if word_to_compare.strip()[0].isupper() or word_to_compare.strip()[0].isdigit():
                                                dataset_appear.append(dataset)
                                                ds_I_list = ['DS-I'] * (dataset_len - 1)
                                                tag_sent = tag_sent.replace(candidate_word, 'DS-B ' + ' '.join(ds_I_list))
                        else:
                            continue


            for method in method_list:
                if method != '':
                    method_len = len(method.split(' '))
                    if len(method) < 3:
                        for word in sent_list:
                            word_to_compare = word.strip(')''('',''.''['']''{''}''?''!'':'';'' ')
                            if method == word_to_compare:
                                if method_len == 1:
                                    if word_to_compare.strip()[0].isupper() or word_to_compare.strip()[0].isdigit():
                                        method_appear.append(method)
                                        tag_sent = tag_sent.replace(word, 'METH-B')
                                elif method_len > 1:
                                    method_appear.append(method)
                                    meth_I_list = ['METH-I'] * (method_len - 1)
                                    tag_sent = tag_sent.replace(word, 'METH-B ' + ' '.join(meth_I_list))
                    else:
                        if sent_len >= method_len:
                            for i in range(0, sent_len - method_len + 1):
                                candidate_word = ' '.join(sent_list[i:i+method_len])
                                if method.lower() in candidate_word.lower():
                                    candidate_word_to_compare = candidate_word.strip(')''('',''.''['']''{''}''?''!'':'';'' ')
                                    a = ngrams(candidate_word_to_compare.strip(), 3)
                                    b = ngrams(method, 3)
                                    c = set(a) & set(b)
                                    dice_sim = (2 * len(c)) / (len(a) + len(b))
                                    if dice_sim >= thres:
                                        if method_len == 1:
                                            if candidate_word_to_compare.strip()[0].isupper() or candidate_word_to_compare.strip()[0].isdigit():
                                                method_appear.append(method)
                                                tag_sent = tag_sent.replace(candidate_word, 'METH-B')
                                        elif method_len > 1:
                                            method_appear.append(method)
                                            meth_I_list = ['METH-I'] * (method_len - 1)
                                            tag_sent = tag_sent.replace(candidate_word, 'METH-B ' + ' '.join(meth_I_list))
                        else:
                            continue

            
            tag_sent_list = tag_sent.split(' ')
            if (len(sent.split())!= len(tag_sent_list)):
                print(tag_sent_list)
                print(sent) 
                print(tag_sent)
                continue
            tag_sent_list_final = []
            for tag in tag_sent_list:
                if tag != 'DS-B' and tag != 'DS-I' and tag != 'METH-B' and tag != 'METH-I':
                    tag_sent_list_final.append('O')
                else: tag_sent_list_final.append(tag)


            tag_sent_final = ' '.join(tag_sent_list_final)

            final_ner_list.append([sent, tag_sent_final])
                #print(sent, tag_sent_final)

    df = pd.DataFrame(final_ner_list, columns=['sent', 'tag'])
    df.to_csv(f'/Data/jiwon/7-classes/230329-ner-dataset-full({thres})-o-tag.csv', encoding='utf8', index=False)

    with open('/Data/jiwon/7-classes/method_appear-full',"wb") as f:
        pickle.dump(method_appear, f)
        
    with open('/Data/jiwon/7-classes/dataset_appear-full',"wb") as f:
        pickle.dump(dataset_appear, f)


def make_dataset_for_thres(thres):
    with open('/Data/jiwon/data/dataset_list_variants', 'rb') as f:
        dataset_list = pickle.load(f)
    dataset_list = list(set(dataset_list))
    dataset_list.sort(key=len, reverse=True)
    
    with open('/Data/jiwon/data/method_list_variants', 'rb') as f:
        method_list = pickle.load(f)
    method_list = list(set(method_list))
    method_list.sort(key=len, reverse=True)


    nlp = spacy.load("en_core_web_md")

    df = pd.read_csv('/Data/jiwon/sample_1000.csv', low_memory = False)
    df_abst = df['sent'].to_list()


    final_ner_list = []

    for sent in tqdm(df_abst):
        sent = sent.replace('\n', '')
        sent = re.sub(' +', ' ', sent)
        sent = sent.strip()
        sent = repr(sent)
        sent = sent[1:-1]
        tag_sent = sent
        sent_list = sent.split(' ')
        sent_len = len(sent_list)

        for dataset in dataset_list:
            if dataset != '':
                dataset_len = len(dataset.split(' '))
                if len(dataset) < 3:
                    for word in sent_list:
                        word_to_compare = word.strip(')''('',''.''['']''{''}''?''!'':'';'' ')
                        if dataset == word_to_compare:
                            if dataset_len == 1:
                                if word_to_compare.strip()[0].isupper() or word_to_compare.strip()[0].isdigit():
                                    tag_sent = tag_sent.replace(word, 'DS-B')
                            elif dataset_len > 1:
                                if word_to_compare.strip()[0].isupper() or word_to_compare.strip()[0].isdigit():
                                    ds_I_list = ['DS-I'] * (dataset_len - 1)
                                    tag_sent = tag_sent.replace(word, 'DS-B ' + ' '.join(ds_I_list))
                else:
                    if sent_len >= dataset_len:
                        for i in range(0, sent_len - dataset_len + 1):
                            candidate_word = ' '.join(sent_list[i:i+dataset_len])
                            if dataset.lower() in candidate_word.lower():
                                candidate_word_to_compare = candidate_word.strip(')''('',''.''['']''{''}''?''!'':'';'' ')
                                a = ngrams(candidate_word_to_compare.strip(), 3)
                                b = ngrams(dataset, 3)
                                c = set(a) & set(b)
                                dice_sim = (2 * len(c)) / (len(a) + len(b))
                                if dice_sim >= thres:
                                    if dataset_len == 1:
                                        if candidate_word_to_compare.strip()[0].isupper() or candidate_word_to_compare.strip()[0].isdigit():
                                            tag_sent = tag_sent.replace(candidate_word, 'DS-B')
                                    elif dataset_len > 1:
                                        if candidate_word_to_compare.strip()[0].isupper() or candidate_word_to_compare.strip()[0].isdigit():                                        
                                            ds_I_list = ['DS-I'] * (dataset_len - 1)
                                            tag_sent = tag_sent.replace(candidate_word, 'DS-B ' + ' '.join(ds_I_list))
                    else:
                        continue


        for method in method_list:
            if method != '':
                method_len = len(method.split(' '))
                if len(method) < 3:
                    for word in sent_list:
                        word_to_compare = word.strip(')''('',''.''['']''{''}''?''!'':'';'' ')
                        if method == word_to_compare:
                            if method_len == 1:
                                if word_to_compare.strip()[0].isupper() or word_to_compare.strip()[0].isdigit():
                                    tag_sent = tag_sent.replace(word, 'METH-B')
                            elif method_len > 1:
                                meth_I_list = ['METH-I'] * (method_len - 1)
                                tag_sent = tag_sent.replace(word, 'METH-B ' + ' '.join(meth_I_list))
                else:
                    if sent_len >= method_len:
                        for i in range(0, sent_len - method_len + 1):
                            candidate_word = ' '.join(sent_list[i:i+method_len])
                            if method.lower() in candidate_word.lower():
                                candidate_word_to_compare = candidate_word.strip(')''('',''.''['']''{''}''?''!'':'';'' ')
                                a = ngrams(candidate_word_to_compare.strip(), 3)
                                b = ngrams(method, 3)
                                c = set(a) & set(b)
                                dice_sim = (2 * len(c)) / (len(a) + len(b))
                                if dice_sim >= thres:
                                    if method_len == 1:
                                        if candidate_word_to_compare.strip()[0].isupper() or candidate_word_to_compare.strip()[0].isdigit():
                                            tag_sent = tag_sent.replace(candidate_word, 'METH-B')
                                    elif method_len > 1:
                                        meth_I_list = ['METH-I'] * (method_len - 1)
                                        tag_sent = tag_sent.replace(candidate_word, 'METH-B ' + ' '.join(meth_I_list))

                    else:
                        continue

        
        tag_sent_list = tag_sent.split(' ')
        if (len(sent.split())!= len(tag_sent_list)):
            print(tag_sent_list)
            print(sent) 
            print(tag_sent)
            continue
        tag_sent_list_final = []
        for tag in tag_sent_list:
            if tag != 'DS-B' and tag != 'DS-I' and tag != 'METH-B' and tag != 'METH-I':
                tag_sent_list_final.append('O')
            else: tag_sent_list_final.append(tag)


        tag_sent_final = ' '.join(tag_sent_list_final)

        final_ner_list.append([sent, tag_sent_final])
            #print(sent, tag_sent_final)

    df = pd.DataFrame(final_ner_list, columns=['sent', 'tag'])
    df.to_csv(f'/Data/jiwon/thres_sample/sample({thres})-inlower.csv', encoding='utf8', index=False)




if __name__ == '__main__':


    #make_dataset(0.1)

    # thres_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # for thres in thres_list:
    #     make_dataset_for_thres(thres)






