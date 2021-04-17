#!/usr/bin/env python
# coding: utf-8

# In[16]:


import logging
import os
import sys
import argparse
import random
import json

import numpy as np
import tqdm
import pickle
import pandas as pd
from matplotlib import pyplot as plt

from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, f1_score

import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW

# from Dataset import Dataset
# sys.path.append(os.path.abspath("../.."))
# from skimage.io import imread, imsave
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")


# In[2]:


def logging_storage(logfile_path):
    logging.basicConfig(filename=logfile_path, filemode='a', level=logging.INFO, format='%(asctime)s => %(message)s')
    logging.info(torch.__version__)
    logging.info(device)


# ### Global Variables

# In[3]:


lr = 2e-5
num_epochs = 1
MAX_LEN = 64
batch_size = 256
dataset = 'clickbaits'


# In[4]:


ending_path = ('%s_%s_%s_%s' %(dataset, MAX_LEN, batch_size, str(lr).replace("-", "")))


# In[5]:


ending_path


# In[6]:


save_model_path = "../models/"
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)

if not os.path.exists("../logs/"):
    os.mkdir("../logs/")

logfile_path = "../logs/" + ending_path


# In[7]:


logging_storage(logfile_path)


# ### Data Loading

# In[38]:


f = open("../data/Clickbait16k/clickbait_data", "r", encoding="utf-8")


# In[39]:


f2 = open("../data/Clickbait16k/non_clickbait_data", "r", encoding="utf-8")


# In[40]:


clickbait_lines = f.readlines()


# In[41]:


non_clickbait_lines = f2.readlines()


# In[42]:


data_clickbait = []
for clickbait in clickbait_lines:
    label = []
    if clickbait != '\n':
        label.append(clickbait.strip())
        label.append(1)
        data_clickbait.append(label)
    else:
        continue

for clickbait in non_clickbait_lines:
    label = []
    if clickbait != '\n':
        label.append(clickbait.strip())
        label.append(0)
        data_clickbait.append(label)
    else:
        continue


# In[43]:


with open('../data/Final_lang.jsonl', 'r') as json_file:
    json_list = list(json_file)


# In[44]:


len(data_clickbait)


# In[45]:


for json_str in json_list:
    result = json.loads(json_str)
    if result['lang'] =='en':
        label = []
        label.append(result['title'])
        label.append(1)
    data_clickbait.append(label)


# In[46]:


df_abc = pd.read_csv('../data/abcnews-date-text.csv')


# In[47]:


for i, row in df_abc.iterrows():
    if i == 650000:
        break
    label = []
    label.append(row[1])
    label.append(0)
    data_clickbait.append(label)


# In[48]:


len(data_clickbait)


# In[50]:


for i in data_clickbait:
    if(len(i) > 2):
        print(i)
    


# # Clickbait label = 1, Non Clickbait title = 0

# In[51]:


df_clickbaits = pd.DataFrame(data_clickbait, columns = ['Title', 'Label'])  


# In[52]:


df_clickbaits


# In[53]:


# for i, row in df_test.iterrows():
#     df_test.at[i, 'input'] = " <cls> " + str(row[2]) + " <sep> " + str(row[3]) + " <cls> "
#     if(i % 100000) == 0 and i:
#         print("Completed: %s" %(i))


# In[54]:


titles = df_clickbaits['Title'].tolist()


# In[55]:


Labels = df_clickbaits['Label'].tolist()


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(titles, Labels, test_size=0.2, random_state=42)


# ### Process Dataset

# In[57]:


tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)


# In[58]:


tokenized_texts_train = [tokenizer.tokenize(inp) for inp in X_train]
input_ids_train = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_train]


# In[59]:


input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


# In[60]:


attention_masks_train = []

for seq in input_ids_train:
    seq_mask = [float(i>0) for i in seq]
    attention_masks_train.append(seq_mask)


# In[61]:


train_inputs = torch.tensor(input_ids_train)
train_labels = torch.tensor(y_train)
train_masks = torch.tensor(attention_masks_train)


# In[62]:


train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


# #### Test

# In[63]:


tokenized_texts_test = [tokenizer.tokenize(inp) for inp in X_test]
input_ids_test = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_test]


# In[64]:


input_ids_test = pad_sequences(input_ids_test, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


# In[65]:


attention_masks_test = []

for seq in input_ids_test:
    seq_mask = [float(i>0) for i in seq]
    attention_masks_test.append(seq_mask)


# In[66]:


test_inputs = torch.tensor(input_ids_test)
test_labels = torch.tensor(y_test)
test_masks = torch.tensor(attention_masks_test)


# In[67]:


test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


# ## Models and Parameters

# In[68]:


model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels = 2)
model = nn.DataParallel(model)
model.to(device)


# In[69]:


logging.info("Model Loaded!")


# In[70]:


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]


# In[71]:


optimizer = AdamW(optimizer_grouped_parameters, lr=lr)


# ## Helper Functions

# In[72]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    labels_flat = labels_flat.cpu().detach().numpy() 
    return np.sum(pred_flat == labels_flat), pred_flat


# In[73]:


def train(i):
    model.train()
    total_loss = 0.0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    train_len = 0
    f_acc = 0

    ## adaptive lr
    optimizer.param_groups[0]['lr'] *= (0.1)**(1/40.)

    logging.info("LR: %s\tEpoch: %s\t" %(optimizer.param_groups[0]['lr'], i)) 
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        if b_labels.size(0) == 1:
            continue
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        pred = outputs[1].detach().cpu().numpy()
        batch_f_acc, pred_flat = flat_accuracy(pred, b_labels)
        f_acc += batch_f_acc
        loss = outputs[0]
        loss.sum().backward()
        optimizer.step()


        labels_flat = b_labels.flatten().cpu().detach().numpy()
        total_actual_label = np.concatenate((total_actual_label, labels_flat))
        total_predicted_label = np.concatenate((total_predicted_label, pred_flat))

#             print(total_actual_label.shape, total_predicted_label.shape)
        total_loss += outputs[0].sum()
        train_len += b_input_ids.size(0)

        if step%100 == 0 and step:
            precision, recall, f1_measure, _ =             precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
            logging.info("Train: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %                         (train_len*100.0/train_inputs.size(0), i, step,                            total_loss/train_len, f_acc*100.0/train_len,                            precision*100., recall*100., f1_measure*100.))

        if torch.cuda.device_count() > 1:
            p = 100
            path = save_model_path + '/e_' + str(i) + "_" + str(p) + ".ckpt"
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)

    precision, recall, f1_measure, _ =     precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
    logging.info("Train: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %                 (train_len*100.0/train_inputs.size(0), i, step,                    total_loss/train_len, f_acc*100.0/train_len,                    precision*100., recall*100., f1_measure*100.))
    return total_loss/train_len


# In[74]:


def test(i):
    model.eval()
    val_len = 0
    total_loss = 0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    f_acc = 0

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.cuda() for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            if b_labels.size(0) == 1:
                continue

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            pred = outputs[1].detach().cpu().numpy()
            batch_f_acc, pred_flat = flat_accuracy(pred, b_labels)
            f_acc += batch_f_acc

            labels_flat = b_labels.flatten().cpu().detach().numpy()
            total_actual_label = np.concatenate((total_actual_label, labels_flat))
            total_predicted_label = np.concatenate((total_predicted_label, pred_flat))

            val_len += b_input_ids.size(0)
            total_loss += outputs[0].sum()

            if step%100 == 0 and step:
                precision, recall, f1_measure, _ =                 precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
                logging.info("Eval: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %                             (val_len*100.0/test_inputs.size(0), i, step,                              total_loss/val_len, f_acc*100.0/val_len,                                precision*100., recall*100., f1_measure*100.))

        precision, recall, f1_measure, _ =         precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
        logging.info("Test: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %                     (val_len*100.0/test_inputs.size(0), i, step,                      total_loss/val_len, f_acc*100.0/val_len,                        precision*100., recall*100., f1_measure*100.))
        return total_actual_label, total_predicted_label


# In[ ]:


train_loss = []
test_loss = []
for i in range(num_epochs):
    train_loss.append(train(i))
    test_loss.append(test(i))

