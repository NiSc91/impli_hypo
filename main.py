
import pandas as pd
import numpy as np
import torch
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

""" Code to prepare dataset which turned out not to be needed
from transformers import RobertaTokenizer
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#len(tokenizer)

cls_token = tokenizer.cls_token
sep_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token
max_input_length = 256

def tokenize_bert(sentence):
    tokens = tokenizer.tokenize(sentence) 
    return tokens

def split_and_cut(sentence):
    tokens = sentence.strip().split(" ")
    tokens = tokens[:max_input_length]
    return tokens

def trim_sentence(sent):
    try:
        sent = sent.split()
        sent = sent[:128]
        return " ".join(sent)
    except:
        return sent

def prepare_data(df_e, df_ne, tokenizer):
    inputs = []
    labels = []

    for i, row in df_e.iterrows():
        premise = row['premise']
        hypothesis = row['hypothesis']
        label = 1 # entailment

        tokens_a = tokenizer.tokenize(premise)
        tokens_b = tokenizer.tokenize(hypothesis)
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.cls_token + tokens_a + tokenizer.sep_token + tokens_b)
        attention_mask = [1] * len(input_ids)

        inputs.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": label})

    for i, row in df_ne.iterrows():
        premise = row['premise']
        hypothesis = row['hypothesis']
        label = 0 # not entailment
        tokens_a = tokenizer.tokenize(premise)
        tokens_b = tokenizer.tokenize(hypothesis)
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.cls_token + tokens_a + tokenizer.sep_token + tokens_b)
        attention_mask = [1] * len(input_ids)

        inputs.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": label})

    return inputs

#inputs = prepare_data(df_e, df_ne)
"""

import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.metrics import classification_report

def process_nli_data(*dfs):
    # Convert dataframes to lists of tuples
    data = []
    true_labels = []
    
    for i, df in enumerate(dfs):
        data.extend(list(zip(df['premise'], df['hypothesis'])))
        true_labels.extend([i] * len(df))
    
    # Load model
    model = CrossEncoder('cross-encoder/nli-deberta-base')
    
    # Predict labels
    scores = model.predict(data)
    label_mapping = ['h1', 'h2', 'h3']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    
    # Get unique labels
    unique_labels = np.unique(true_labels)
    
    # Map labels to integers
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    model_labels = np.array([label_to_int[label] for label in labels])
    
    # Model evaluation
    report = classification_report(true_labels, model_labels)
    
    return report

# Define dataframes for multiple datasets
dataset1_h1 = pd.read_csv('metaphors/manual_e1.tsv', sep='\t', names=['premise', 'hypothesis'])
dataset1_h2 = pd.read_csv('metaphors/manual_ne1.tsv', sep='\t', names=['premise', 'hypothesis'])

dataset2_df = pd.read_csv('hyperboles/hypo_nli.csv')
# Transform dataset 2 into a two-column structure
dataset2_h1 = dataset2_df[['premise', 'entailment']].copy()
dataset2_h1.columns = ['premise', 'hypothesis']
dataset2_h2 = dataset2_df[['premise', 'contradiction']].copy()
dataset2_h2.columns = ['premise', 'hypothesis']
dataset2_h3 = dataset2_df[['premise', 'neutral']].copy()
dataset2_h3.columns = ['premise', 'hypothesis']

# Process and evaluate dataset 1 (2 classes)
dataset1_report = process_nli_data(dataset1_h1, dataset1_h2)
print("Dataset 1 report:")
print(dataset1_report)

# Process and evaluate dataset 3 (2 classes)
dataset2_report = process_nli_data(dataset2_h1, dataset2_h2, dataset2_h3)
print("Dataset 2 report:")
print(dataset2_report)
