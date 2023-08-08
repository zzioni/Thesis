import pandas as pd
import pickle
from tqdm import tqdm
import random
import re
import numpy as np

  
def random_RD():
    df = pd.read_csv('/Data/jiwon/230329-dataset-for-NERtraining-full.csv', low_memory = False)

    sents = df['sent'].to_list()
    tags = df['tag'].to_list()    

    final_list = []

    for sent, tag in tqdm(zip(sents, tags)):

        sent_list = sent.split(' ')
        tag_list = tag.split(' ')

        count = len(sent_list)
        per = int(count * 0.1)

        num_list = list(range(count))

        for i, tag_item in enumerate(tag_list):
            if tag_item != 'O':
                num_list.remove(i)

        for i in range(1):
            sent_list = sent.split(' ')
            tag_list = tag.split(' ')            
            random_index = random.sample(num_list, per)
            random_index.sort(reverse=True)
            for index in random_index:
                del sent_list[index]
                del tag_list[index]
            if len(sent_list) != len(tag_list):
                print(sent)
            
            final_sent = ' '.join(sent_list)
            final_tag = ' '.join(tag_list)
            final_list.append([final_sent, final_tag])

    df = pd.DataFrame(final_list, columns=['sent', 'tag'])
    df.to_csv('230329-sent-augmentation.csv', encoding='utf8', index=False)


  
def count():
    df = pd.read_csv('/Data/jiwon/230329_FINAL_NER_DATASET.csv', low_memory = False)
    sents = df['sent'].to_list()
    tag_sents = df['tag'].to_list()

    ds_count = 0
    meth_count = 0
    o_count = 0

    ds_meth_sent = []
    ds_sent = []
    meth_sent = []

    for sent, tags in zip(sents, tag_sents):
        ds_count += tags.count('DS-B')
        meth_count += tags.count('METH-B')
        o_count += tags.count('O')

        if ('DS-B' in tags) and ('METH-B') in tags:
            if len(tags.split()) >= 5:
                ds_meth_sent.append([sent, tags])
        elif 'DS-B' in tags:
            if len(tags.split()) >= 5:
                ds_sent.append([sent, tags])
        elif 'METH-B' in tags:
            if len(tags.split()) >= 5:
                meth_sent.append([sent, tags])

    return ds_meth_sent, ds_sent, meth_sent



def ner_data_aumentation():
    final_list = []

    with open('/Data/jiwon/data/dataset_list', 'rb') as f:
        dataset_list = pickle.load(f)
        dataset_list = list(set(dataset_list))
    with open('/Data/jiwon/data/method_list', 'rb') as f:
        method_list = pickle.load(f)
        method_list.remove('Multi Loss ( BCE Loss + Focal Loss )  + Dice Loss')
        method_list.append('Multi Loss + Dice Loss')    
        method_list = list(set(method_list))    
        

    with open('/Data/jiwon/dataset_method_appear/dataset_appear-full', 'rb') as f:
        dataset_appear_list = pickle.load(f)
    with open('/Data/jiwon/dataset_method_appear/method_appear-full', 'rb') as f:
        method_appear_list = pickle.load(f)   


    dataset_list_to_be = list(set(dataset_list) - set(dataset_appear_list))
    method_list_to_be = list(set(method_list) - set(method_appear_list))

    ds_meth_sent, ds_sent, method_sent = count()
    
    dataset_sent = ds_meth_sent + ds_sent

    for dataset in dataset_list:
        
        random_ds_sent = random.sample(dataset_sent, 20)
        for ds_sent in random_ds_sent:
            sent_list = ds_sent[0].split(' ')
            tag_list = ds_sent[1].split(' ')

            if len(sent_list) >=5:

                tag_list_np = np.array(tag_list)

                start_loc_list = np.where(tag_list_np == 'DS-B')[0]
                start_loc_list = start_loc_list.tolist()

                start_loc = random.choice(start_loc_list)
                end_loc = start_loc

                if end_loc+1 != len(tag_list):
                    while tag_list[end_loc+1] == 'DS-I':
                        end_loc +=1
                        if end_loc+1 == len(tag_list):
                            break
                
                del sent_list[start_loc:end_loc+1]
                del tag_list[start_loc:end_loc+1]

                dataset_name_len = len(dataset.split(' '))
                sent_list.insert(start_loc, dataset)
                if dataset_name_len == 1:
                    tag_list.insert(start_loc, 'DS-B')
                elif dataset_name_len > 1:
                    ds_I_list = ['DS-I'] * (dataset_name_len -1)
                    add_tag =  'DS-B ' + ' '.join(ds_I_list)
                    tag_list.insert(start_loc, add_tag)

                final_sent = ' '.join(sent_list)
                fianl_tag = ' '.join(tag_list)

                final_list.append([final_sent, fianl_tag])


    for method in method_list:
        random_meth_sent = random.sample(method_sent, 10)
        for meth_sent in random_meth_sent:
            sent_list = meth_sent[0].split(' ')
            tag_list = meth_sent[1].split(' ')
            if len(sent_list) >=3:

                tag_list_np = np.array(tag_list)

                start_loc_list = np.where(tag_list_np == 'METH-B')[0]
                start_loc_list = start_loc_list.tolist()
                start_loc = random.choice(start_loc_list)
                end_loc = start_loc
                
                if end_loc+1 != len(tag_list):
                    while tag_list[end_loc+1] == 'METH-I':
                        end_loc +=1
                        if end_loc+1 == len(tag_list):
                            break
                
                del sent_list[start_loc:end_loc+1]
                del tag_list[start_loc:end_loc+1]

                method_name_len = len(method.split(' '))
                sent_list.insert(start_loc, method)
                if method_name_len == 1:
                    tag_list.insert(start_loc, 'METH-B')
                elif method_name_len > 1:
                    meth_I_list = ['METH-I'] * (method_name_len -1)
                    add_tag =  'METH-B ' + ' '.join(meth_I_list)
                    tag_list.insert(start_loc, add_tag)

                final_sent = ' '.join(sent_list)
                fianl_tag = ' '.join(tag_list)

                final_list.append([final_sent, fianl_tag])

    df = pd.DataFrame(final_list, columns=['sent', 'tag'])
    df.to_csv('230329-ner-dataset-augmentation.csv', encoding='utf8', index=False)




if __name__ == '__main__':
    #count()
    #random_RD()
    #ner_data_aumentation()



