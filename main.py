import pdb
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
    """ Function takes a dataframe, wherein each one corresponds to an NLI pair.
    If 2 classes, the order is: Entailment, non-entailment
    If 3 classes, the order is: Contradiction, entailment, neutral
    """
    # Convert dataframes to lists of tuples
    data = []; true_labels = []; num_classes = len(dfs)
    
    for i, df in enumerate(dfs):
        data.extend(list(zip(df['premise'], df['hypothesis'])))
        true_labels.extend([i] * len(df))
    
    #pdb.set_trace()
    # Load model
    model = CrossEncoder('cross-encoder/nli-deberta-base')
    scores = model.predict(data)
    # Get attention weights
    attention_weights = model.attention_weights(data, visualize=True)


    #Convert scores to labels
    #Assumes the order ['contradiction', 'entailment', 'neutral']
    model_labels = scores.argmax(axis=1)

    if num_classes == 2:
        # Re-map model labels to two classes (non-entailment will be 0)
        model_labels = np.array([0 if l == 2 else l for l in model_labels])
    
    # Model evaluation
    report = classification_report(true_labels, model_labels)
    
    return report

# Define dataframes for multiple datasets
dataset1_entailment = pd.read_csv('metaphors/manual_e.tsv', sep='\t', names=['premise', 'hypothesis'])
dataset1_non_entailment = pd.read_csv('metaphors/manual_ne.tsv', sep='\t', names=['premise', 'hypothesis'])

dataset2_df = pd.read_csv('hyperboles/hypo_NLI_V1.csv')
dataset2_df.dropna(subset='neutral', inplace=True)

# Transform dataset 2 into a two-column structure
dataset2_entailment = dataset2_df[['premise', 'entailment']].copy()
dataset2_entailment.columns = ['premise', 'hypothesis']
dataset2_contradiction = dataset2_df[['premise', 'contradiction']].copy()
dataset2_contradiction.columns = ['premise', 'hypothesis']
dataset2_neutral = dataset2_df[['premise', 'neutral']].copy()
dataset2_neutral.columns = ['premise', 'hypothesis']

# Process and evaluate dataset 1 (2 classes)
dataset1_report = process_nli_data(dataset1_non_entailment, dataset1_entailment)
print("Dataset 1 report:")
print(dataset1_report)

# Process and evaluate dataset 2 (2 classes)
dataset2_report = process_nli_data(dataset2_contradiction, dataset2_entailment, dataset2_neutral)
print("Dataset 2 report:")
print(dataset2_report)
