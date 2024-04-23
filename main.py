import pandas as pd
import numpy as np
import torch
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

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

# Load and prepare datasets (two tsv-files in pandas corresponding to entailment and non-entailment for an NLI task)
# Read the files metaphors/manual_e.tsv and metaphors/manual_ne.tsv
df_e = pd.read_csv('metaphors/manual_e.tsv', sep='\t', names=['premise', 'hypothesis'])
df_ne = pd.read_csv('metaphors/manual_ne.tsv', sep='\t', names=['premise', 'hypothesis'])

df_e.head()

# Convert dataframe  to a list with the two columns as pairs of tuples
data_e = list(df_e.apply(lambda x: (x['premise'], x['hypothesis']), axis=1))
data_ne = list(df_ne.apply(lambda x: (x['premise'], x['hypothesis']), axis=1))
# Concatenate data lists
data = data_e+data_ne

print(len(data))

# Make labels (numpy array)
true_labels = np.array([1] * len(data_e) + [0] * len(data_ne))

## Load model
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/nli-deberta-base')
scores = model.predict(data)

#Convert scores to labels
label_mapping = ['non-entailment', 'entailment', 'neutral']
labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

# Roll neutral and contradiction labels into one non-entailment label
model_labels = np.array([1 if label == 'entailment' else 0 for label in labels])

# Model evaluation
from sklearn.metrics import classification_report
print(classification_report(true_labels, model_labels))
