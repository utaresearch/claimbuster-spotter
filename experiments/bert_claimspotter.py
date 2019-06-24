#!/usr/bin/env python
# coding: utf-8

import os
import json

import logging
import numpy as np
import pandas as pd

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.optimization import BertAdam

from tqdm import trange
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 10
MAX_SEQ_LENGTH=200
GRADIENT_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 100
LEARNING_RATE = 5e-5
WARMUP_PROPORTION = 0.1
PATIENCE = 3

OUTPUT_DIR = "/tmp/"
MODEL_FILE_NAME = "pytorch_model.bin"

BERT_MODEL = "bert-base-uncased"
#BERT_MODEL = "bert-base-cased"
#BERT_MODEL = "bert-large-uncased"

#to make sure this code runs on any machine we'll let PyTorch determine whether a GPU is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Each model comes with its own tokenizer. This tokenizer splits texts into word pieces. 
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def convert_examples_to_features(example_texts, example_labels, label2idx, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):
        tokens = tokenizer.tokenize(text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
            
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
            
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label2idx[label]
        if verbose and ex_index == 0:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label:" + str(label) + " id: " + str(label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def get_data_loader(features, max_seq_length, batch_size): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader



def evaluate(model, dataloader):

    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        #outputs = np.argmax(logits, axis=1)
        outputs_temp = logits.cpu().numpy()
        outputs = np.argmax(outputs_temp, axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels



def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x



def train_classifier(train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels):
    
    TARGET_NAME_PATH = os.path.join(os.path.expanduser("~"), "target_names.json")
    target_names = list(set(train_labels))
    with open(TARGET_NAME_PATH, "w") as o:
        json.dump(target_names, o)
    
    label2idx = {label: idx for idx, label in enumerate(target_names)}
    #print(label2idx)
    
    train_features = convert_examples_to_features(train_texts, train_labels, label2idx, MAX_SEQ_LENGTH, tokenizer, verbose=0)
    dev_features = convert_examples_to_features(dev_texts, dev_labels, label2idx, MAX_SEQ_LENGTH, tokenizer)
    test_features = convert_examples_to_features(test_texts, test_labels, label2idx, MAX_SEQ_LENGTH, tokenizer)
    
    train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, BATCH_SIZE)
    dev_dataloader = get_data_loader(dev_features, MAX_SEQ_LENGTH, BATCH_SIZE)
    test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE)
    
    #A full BERT model consists of a common, pretrained core, and an extension on top that depends on 
    #the particular NLP task. As we're looking at text classification, we're going to use the 
    #pretrained BERT model with a final layer for sequence classification on top.
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = len(label2idx))
    model.to(device)
    
    num_train_steps = int(len(train_texts) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    
    optimizer = BertAdam(optimizer_grouped_parameters, LEARNING_RATE,
                     warmup=WARMUP_PROPORTION, t_total=t_total)
    
    global_step = 0
    model.train()
    loss_history = []
    
    for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            
            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                lr_this_step = LEARNING_RATE * warmup_linear(global_step/t_total, WARMUP_PROPORTION)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        
        dev_loss, _, _ = evaluate(model, dev_dataloader)
        print("Loss history:", loss_history)
        print("Dev loss:", dev_loss)
        if len(loss_history) == 0 or dev_loss < min(loss_history):
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
        
        if len(loss_history) > 0 and dev_loss > max(loss_history[-PATIENCE:]): 
            print("No improvement on development set. Finish training.")
            break
        
        loss_history.append(dev_loss)
    
    _, train_correct, train_predicted = evaluate(model, train_dataloader)
    _, dev_correct, dev_predicted = evaluate(model, dev_dataloader)
    _, test_correct, test_predicted = evaluate(model, test_dataloader)

    print("Training performance:", precision_recall_fscore_support(train_correct, train_predicted, average="micro"))
    print("Development performance:", precision_recall_fscore_support(dev_correct, dev_predicted, average="micro"))
    print("Test performance:", precision_recall_fscore_support(test_correct, test_predicted, average="micro"))
    
    print(classification_report(test_correct, test_predicted, target_names=target_names))



def test_classifier(test_texts, test_labels):
     
    output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
    TARGET_NAME_PATH = os.path.join(os.path.expanduser("~"), "target_names.json")
    with open(TARGET_NAME_PATH) as i:
        target_names = json.load(i)
    label2idx = {label: idx for idx, label in enumerate(target_names)}
    test_features = convert_examples_to_features(test_texts, test_labels, label2idx, MAX_SEQ_LENGTH, tokenizer)
    test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE)
    
    model_state_dict = torch.load(output_model_file)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, state_dict=model_state_dict, num_labels = len(target_names))
    model.to(device)
    
    model.eval()
    _, test_correct, test_predicted = evaluate(model, test_dataloader) 
    
    print("Test performance:", precision_recall_fscore_support(test_correct, test_predicted, average="micro"))
    print(classification_report(test_correct, test_predicted, target_names=target_names))
    
