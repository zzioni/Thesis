import pandas as pd
import json
from tqdm import tqdm
import pickle

def dataset_parsing():
    path = 'D:/thesis/paperswithcode_dataset'
    file = open(path, encoding='utf8')
    json_data = json.load(file)

    final_list = []
    modality_full = []
    for i, data in tqdm(enumerate(json_data)):

        name = data['name']
        full_name = data['full_name']
        modalities = data['modalities']
        variants_list = data['variants']
        variants = '|||'.join(variants_list)
        for modality in modalities:
            if 'medical' in modality.lower() or 'images' in modality.lower() :
                final_list.append([str(name), str(full_name), str(variants)])
                break



    df = pd.DataFrame(final_list, columns=['name', 'full_name', 'variants'])
    df.to_csv('dataset-all-variants.csv', encoding='utf8', index=False)


def method_parsing():
    path = 'D:/thesis/paperswithcode_method'
    file = open(path, encoding='utf8')
    json_data = json.load(file)

    print(len(json_data))

    final_list =[]

    for i, data in tqdm(enumerate(json_data)):

        name = data['name']
        full_name = data['full_name']

        final_list.append([name, full_name])

    df = pd.DataFrame(final_list, columns=['name', 'full_name'])
    df.to_csv('method.csv', encoding='utf8', index=False)


def method_parsing():
    path = 'D:/thesis/paperswithcode_paper'
    file = open(path, encoding='utf8')
    json_data = json.load(file)

    print(len(json_data))

    final_list =[]

    for i, data in tqdm(enumerate(json_data)):

        arxiv_id = data['arxiv_id']
        title = data['title']
        try :
            abstract = data['abstract'].strip().replace('\n','')
        except: pass
        tasks_list = data['tasks']
        tasks = '|||'.join(tasks_list)
        date = data['date']

        final_list.append([arxiv_id, title, abstract, tasks, date])
        #final_list.append([tasks])

    df = pd.DataFrame(final_list, columns=['arxiv_id', 'title', 'abstract', 'tasks', 'date'])
    df_dop_row = df.dropna(axis=0)
    df_dop_row.to_csv('paper.csv', encoding='utf8', index=False)
  



def dataset_list_maker():
    dataset_path = 'entity_list/dataset-all-variants.csv'
    dataset_data = pd.read_csv(dataset_path)
    dataset_list = []


    dataset_variants = dataset_data['variants'].to_list()
    dataset_variants_final = []
    for variants in dataset_variants:
        try:
            variants = variants.split('|||')
        except:
            continue
        dataset_variants_final = dataset_variants_final + variants
    dataset_list = list(set(dataset_data['name'].to_list() + dataset_variants_final + dataset_data['full_name'].to_list()))
    dataset_list_final = []

    for dataset in dataset_list:
        dataset_list_final.append(str(dataset))
    dataset_list_final.sort(reverse=True)
    print(len(dataset_list_final))
    with open('entity_list/dataset_list_variants_all', "wb") as f:
        pickle.dump(dataset_list_final, f)

def method_list_maker():
  
  method_path = 'method.csv'
  method_data = pd.read_csv(method_path)
  
  # method_list = []
  #
  # for method in method_data['name'].to_list():
  #     method_list.append(method.lower())
  
  method_list = list(set(method_data['name'].to_list() + method_data['full_name'].to_list()))
  method_list_final = []
  for method in method_list:
      method_list_final.append(str(method))
  method_list_final.sort(reverse=True)
  print(len(method_list_final))
  with open('method_list_variants',"wb") as f:
      pickle.dump(method_list_final, f)



if __name__ == '__main__':
    #dataset_parsing()
  
