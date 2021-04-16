import logging
import os
import sys
import argparse
import random

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    
## GENERAL Prameters
MAX_LEN = 64
batch_size = 256

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels = 2)

model.load_state_dict(torch.load("./models/e_0_100.ckpt", map_location=device))

model.to(device)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

title = sys.argv[1]

clickbaits = []

clickbaits.append(title)

def Dataloader(titles):
    tokenized_texts = [tokenizer.tokenize(titles) for titles in titles]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    validation_inputs = torch.tensor(input_ids)
    validation_masks = torch.tensor(attention_masks)

    validation_data = TensorDataset(validation_inputs, validation_masks)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return validation_dataloader
    
Validation_Dataloader = Dataloader(clickbaits)

def eval(validation_dataloader):
    model.eval()
    val_len = 0
    total_loss = 0
    predictions = []
    with torch.no_grad():
        for step, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            preds = outputs[0].detach().cpu().numpy()
            pred_flat = np.argmax(preds, axis=1).flatten()
            predictions.append(pred_flat)
            if step%20 and step:
                print(step)
    return predictions
    
preds = eval(Validation_Dataloader)

f= open("output.txt","w+")
for i in preds:
    f.write( str(i))