import xml.etree.ElementTree as ET
import pandas as pd
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
import pickle
import random
import argparse



def xml_parsing():

    pkl_name = 'no-list.pkl'
    csv_name = 'pmc_fulltext.csv'

    print(pkl_name)
    print(csv_name)

    no_list = []
    final_list = []

    list_path = ''
    with open(list_path, "r", encoding="utf8") as f:
        lines = f.readlines()
    pmc_list = [l.replace('\n','') for l in lines]
    print(len(pmc_list))


    for ID in tqdm(pmc_list):
        pmc = ID
      
        if len(pmc) == 7:
            start_num = '0'+pmc[0]

        elif len(pmc) == 8:
            start_num = pmc[:2] 
            
        else: start_num = '00'

        pmc_dir_author = '/Data/2022.PMC/author/' + 'PMC0' + start_num + 'xxxxxx/PMC' + pmc + '.xml'
        pmc_dir_comm = '/Data/2022.PMC/oa_comm/' + 'PMC0' + start_num + 'xxxxxx/PMC' + pmc + '.xml'
        pmc_dir_noncomm = '/Data/2022.PMC/oa_noncomm/' + 'PMC0' + start_num + 'xxxxxx/PMC' + pmc + '.xml'
        pmc_dir_other = '/Data/2022.PMC/oa_other/' + 'PMC0' + start_num + 'xxxxxx/PMC' + pmc + '.xml'

        if os.path.isfile(pmc_dir_author):
            pmc_dir = pmc_dir_author
        elif os.path.isfile(pmc_dir_comm):
            pmc_dir = pmc_dir_comm
        elif os.path.isfile(pmc_dir_noncomm):
            pmc_dir = pmc_dir_noncomm
        elif os.path.isfile(pmc_dir_other):
            pmc_dir = pmc_dir_other
        else: 
            no_list.append(pmc)
            continue
    
        xml_data = open(pmc_dir, "r", encoding="utf-8").read()
    
        soup = BeautifulSoup(xml_data, 'lxml')

        front = soup.find('front')
        try:
            title = front.find('title-group').text.replace('\n', '')
        except: title = ''

        try: 
            abstract = front.find('abstract').text.replace('\n', '').replace('\t','')     
        except: abstract = ''

        try:
            year = front.find('year').text
        except : year = ''
        
        try: 
            month = front.find('month').text
        except: year =''

        try:
            day = front.find('day').text
        except: year =''

        full_text = ''

        article = soup.find('body')
        for element in article.find_all(['p', 'sec']):
            try: 
                title_text = element.find('title').text.lower()
            except : title_text =''
            if 'acknowledgment' in title_text or 'Supplementary Material' in title_text or 'footnotes' in title_text or 'contributor information' in title_text or 'reference' in title_text or 'abstract' in title_text:
                pass
            else:      
                content = element.text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
                full_text = full_text + ' ' + content
        print(full_text)
        final_list.append([pmc, title, year, month, day, abstract, full_text])


    print('final', len(final_list))
    print('no', len(no_list))


    with open(pkl_name,"wb") as f:
        pickle.dump(no_list, f)

    df = pd.DataFrame(final_list, columns=['pmc', 'title', 'year', 'month', 'day', 'abstract', 'full_text'])
    df.to_csv(csv_name, index=False)



if __name__ == '__main__':
    xml_parsing()





   
