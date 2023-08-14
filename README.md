# Biomedical document classification based on Named Entity Recognition: Focusing on dataset and method entities
This study proposes an architecture that combines relevant entity information with a BERT-based document classification model. Dataset and method, which are treated as key entities that carry important information of scientific literature in the domain of computer science and artificial intelligence, are adopted as entity information. The document classification models suggested in this study are EnBERT and SenBERT, which employs entity information in two different ways to improve the language representation of a document. While EnBERT enhances the language representation of a paper by merging abstract with the extracted dataset and method entities, SenBERT applies the embedding of the sentence containing these entities to reflect the contextual information in which the entities appeared.

The overall research process is as follows: The training dataset for each model is constructed by collecting papers related to computer vision tasks from PubMed Central (PMC), the full-text archive of biomedical journals. In the process of training NER model, precision, recall, and F1 scores are measured for each epoch, and the model with the highest performance is finally selected. The document classification experiment is designed to compute the macro F1 score of EnBERT, SenBERT, and the baseline model for comparative analysis. In the Additional Experiment, performance evaluation isconducted on REnBERT and RSenBERT, which are models that respectively use randomly selected words and sentences from abstracts as additional information. The results are compared with the models proposed in this study-EnBERT and SenBERT.
<br></br>
## 1. Workflow
<br></br>
<p align="center"><img src=https://github.com/zzioni/Thesis-Master/assets/106359887/0a95ca55-a552-4a22-a8ea-b120e96bb673"  width="800" style="margin:auto; display:block;"></p>
<br></br>

## 2. Data collection and preprocessing
### 1) collecting biomedical from Pubmed Central
**(1) Making a pmc list:** data_collection/pmc_xml_parsing.py

**(2) XML parsing:** ./data_collection/pmc_xml_parsing.py
<br></br>
### 2) collecting Dataset & Method list
**(1) Json parsing:** ./data_collection/paperswithcode_json_parsing.py
<br></br>

## 3. Building the NER dataset
**(1) Automatic NER tagging:** ./making_ner_dataset/auto_entity_tagging.py

**(2) Calculating f1-score for selecting thershold of automatic tagging:** ./making_ner_dataset/compute_f1scoreFORthres

**(3) Data aumentation:** ./making_ner_dataset/dataset_augmentation.py

**(4) others:** ./making_ner_dataset/postprocessing.py
<br></br>

## 4. Document classification
**(1) Baseline-model** ./document_classification/Baseline-kfold.py

**(2) EnBERT** ./document_classification/EnBERT-kfold.py
<br></br>
<p align="center"><img src=https://github.com/zzioni/Thesis/assets/106359887/be2380a9-e46c-468e-aecf-5bcea2e3cf5a"  width="600" style="margin:auto; display:block;"></p>
<br></br>

**(3) SenBERT** ./document_classification/SenBERT-kfold.py
<br></br>
<p align="center"><img src=https://github.com/zzioni/Thesis/assets/106359887/00b8cabb-e34a-4b79-a696-4f76078b7bcd"  width="600" style="margin:auto; display:block;"></p>
<br></br>
