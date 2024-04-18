import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam, AdamW, RMSprop
from tqdm import tqdm
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--WD", default=0.01, type=float)
parser.add_argument("--LR", default=5e-5, type=float)
parser.add_argument("--NUM", default=1, type=float)
args = parser.parse_args()

log_path = f'{args.LR}/entity-log/'
os.makedirs(log_path)

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

labels = {'registration':0,
          'classification':1,
          'detection':2,
          'segmentation':3,
          'enhancement':4,
          'reconstruction':5,
          'CAD':6
          }


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['task']]
        self.texts = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['abstract']]
        self.tagged_entitiy = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['ner_entity']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_texts_tag(self, idx):
        # Fetch a batch of inputs
        return self.tagged_entitiy[idx]

    def __getitem__(self, idx):
        batch_tagged_sent = self.get_batch_texts_tag(idx)
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y, batch_tagged_sent


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.linear = nn.Linear(1536, 7)

    def forward(self, input_id, mask, tagged_mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        tagged_hidden, tagged_pooled_output = self.bert(input_ids= input_id, attention_mask=tagged_mask,return_dict=False)
        tagged_hidden =tagged_hidden[:,1:-1,:]
        tagged_hidden_mean = torch.mean(tagged_hidden, dim=1)
        concat_pooled_output = torch.cat([pooled_output, tagged_hidden_mean], dim=1)  
        linear_output = self.linear(concat_pooled_output)

        return linear_output


def train(data, learning_rate, epochs):

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(data)):


        print(f'-----------------------------FOLD {fold}----------------------------')

        model = BertClassifier()

        f_train = open(f"{log_path}/output-train-{fold}.txt", "a", encoding='utf8')
        f_predict = open(f"{log_path}/output-predict-{fold}.txt", "a", encoding='utf8')


        train_id = train_ids.tolist()
        val_id = valid_ids.tolist()

        train_data = data.copy()
        val_data = data.copy()


        train_data.drop(val_id, axis=0, inplace=True)
        val_data.drop(train_id, axis=0, inplace=True)  
        
        train, val = Dataset(train_data), Dataset(val_data)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=48, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=80)

        classes = ['registration','classification','detection','segmentation', 'enhancement', 'reconstruction', 'CAD']
        #classes = ['registration','classification','detection','segmentation']
        yclasses =[]

        for label_task in train_data['task'].values.tolist():
            yclasses.append(label_task)


        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        
        classweight = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=yclasses)
        classweight = torch.Tensor(classweight)
        classweight = classweight.to(device, dtype = torch.float)


        criterion = nn.CrossEntropyLoss(weight=classweight)
        optimizer = Adam(model.parameters(), lr= learning_rate)   
        #optimizer = AdamW(model.parameters(), lr= learning_rate, weight_decay=args.WD)
        #optimizer = RMSprop(model.parameters(), lr= learning_rate) 


        if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
            


        for epoch_num in range(epochs):
            print(f'---------------------------------------------EPOCH:{epoch_num+1}---------------------------------------------')
            f_train.write(f'---------------------------------------------EPOCH:{epoch_num+1}---------------------------------------------\n')
            f_predict.write(f'---------------------------------------------EPOCH:{epoch_num+1}---------------------------------------------\n')

            tr_loss, tr_accuracy = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            tr_preds, tr_labels = [], []
            val_loss, val_accuracy = 0, 0
            nb_val_examples, nb_val_steps = 0, 0
            val_preds, val_labels = [], []

            for train_input, train_label, tagged_input in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                tagged_mask = tagged_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)


                output = model(input_id, mask, tagged_mask)
                
                loss = criterion(output, train_label)

                flattened_targets = train_label.view(-1)
                active_logits = output.view(-1, len(classes))  

                flattened_predictions = torch.argmax(active_logits, axis=1)
                active_accuracy = train_label.view(-1) != -100   



                labels = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)
                
                tr_loss += loss.mean().item()
                nb_tr_steps += 1
                nb_tr_examples += train_label.size(0)        
                
                tr_labels.extend(labels.cpu().tolist())
                tr_preds.extend(predictions.cpu().tolist())        

                tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                
                tr_accuracy += tmp_tr_accuracy        
                #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)
                
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()
            

            epoch_loss = tr_loss / nb_tr_steps
            tr_accuracy = tr_accuracy / nb_tr_steps
            print(f'epoch_loss={epoch_loss}, tr_accuracy={tr_accuracy}')
            f_train.write(f'epoch_loss={epoch_loss}, tr_accuracy={tr_accuracy}\n')
            #print(classification_report(tr_labels, tr_preds, target_names=classes, zero_division=0))
            f_train.write(classification_report(tr_labels, tr_preds, target_names=classes, zero_division=0))
                


            with torch.no_grad():

                for val_input, val_label, val_tagged_input in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    tagged_mask = val_tagged_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask, tagged_mask)


                    loss = criterion(output, val_label)

                    flattened_targets = val_label.view(-1)
                    active_logits = output.view(-1, len(classes))  

                    flattened_predictions = torch.argmax(active_logits, axis=1)
                    active_accuracy = val_label.view(-1) != -100   


                    labels = torch.masked_select(flattened_targets, active_accuracy)
                    predictions = torch.masked_select(flattened_predictions, active_accuracy)
                    
                    val_loss += loss.mean().item()
                    nb_val_steps += 1
                    nb_val_examples += val_label.size(0)        
                    
                    val_labels.extend(labels.cpu().tolist())
                    val_preds.extend(predictions.cpu().tolist())        

                    tmp_val_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                    
                    val_accuracy += tmp_val_accuracy        
                

                epoch_loss = val_loss / nb_val_steps
                val_accuracy = val_accuracy / nb_val_steps

                print(f'epoch_loss={epoch_loss}, val_accuracy={val_accuracy}')
                print(classification_report(val_labels, val_preds, target_names=classes, zero_division=0))
                f_predict.write(f'epoch_loss={epoch_loss}, val_accuracy={val_accuracy}\n')
                f_predict.write(classification_report(val_labels, val_preds, target_names=classes, zero_division=0))
        f_train.close()
        f_predict.close()
                  

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

input_dir = '/data0/jiwon/Classification/230417-ner-for-classification-abst(dupli).csv'
data = pd.read_csv(input_dir, encoding="utf-8")

data = data.dropna()

df = data[['abstract', 'task', 'ner_entity']]
df = df.reset_index()

df = df.applymap(str)


df = shuffle(df).reset_index(drop=True)


fold_count = 5
kfold = KFold(n_splits=fold_count, random_state=0, shuffle=True)


LR = 5e-6
LR = args.LR
MAX_GRAD_NORM = 5
EPOCHS =15


train(df, LR, EPOCHS)
