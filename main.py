import pdb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample, losses
# Arguments used for new methods of sentence_transformers to train/fine-tune (not implemented yet)
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
import torch.nn as nn

class NLIDataset(Dataset):
    def __init__(self, premise_list, hypothesis_list, labels):
        self.premise_list = premise_list
        self.hypothesis_list = hypothesis_list
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        premise = self.premise_list[idx]
        hypothesis = self.hypothesis_list[idx]
        label = self.labels[idx]
        return InputExample(texts=[premise, hypothesis], label=label)

def depricated_train(model, train_dataloader, output_path="models/", num_epochs=20):
    # Define the loss function
    #train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2) ## Problem with .get_sentence_embedding_dim attribute, which does not exist for CrossEncoder models
    
    model.fit(train_dataloader, epochs=10, warmup_steps=100, output_path="finetuned_models/NLIDebertaImpli")
    
    return model

def new_train(model, train_dataloader):
    
    ## Specify training loss
    loss = losses.CoSENTLoss(model)

    ## Specify training arguments
    args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/NLIDebertaBaseImpli",
    # Optional training parameters:
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    #batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
    )
    
    ## Train model
    trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataloader,
    loss=loss)
    
    trainer.train()
    model.save_pretrained("models/NLIDebertaBaseImpli/final")
    
    return model

def train_pytorch(model, train_dataloader, num_epochs=20, eval_dataloader=None):
    # Specify optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_function = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_function(outputs.logits, batch["labels"])
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {average_loss:.4f}")

    # Evaluate the model if eval_dataloader is provided
    if eval_dataloader is not None:
        model.eval()
        eval_losses = []
        eval_labels = []
        eval_preds = []

        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                eval_losses.append(loss_function(outputs.logits, batch["labels"]).item())
                eval_labels.extend(batch["labels"].cpu().numpy())
                eval_preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())

        eval_loss = np.mean(eval_losses)
        eval_report = classification_report(eval_labels, eval_preds)
        print(f"Evaluation Loss: {eval_loss:.4f}")
        print("Evaluation Report:")
        print(eval_report)

    return model

def test(model, data, true_labels):
    """ Function takes the model and dataframes, wherein each one corresponds to an NLI pair.
    If 2 classes, the order is: Entailment, non-entailment
    If 3 classes, the order is: Contradiction, entailment, neutral
    """
    
    scores = model.predict(data)
    # Get attention weights
    #attention_weights = model.attention_weights(data, visualize=True)

    #Convert scores to labels
    #Assumes the order ['contradiction', 'entailment', 'neutral']
    model_labels = scores.argmax(axis=1)
    num_classes = len(data)

    if num_classes == 2:
        # Re-map model labels to two classes (non-entailment will be 0)
        model_labels = np.array([0 if l == 2 else l for l in model_labels])
    
    # Model evaluation
    report = classification_report(true_labels, model_labels)
    
    return report

def convert_data(*dfs):
    # Convert dataframes to lists of tuples
    data = []; true_labels = []; num_classes = len(dfs)
    
    for i, df in enumerate(dfs):
        data.extend(list(zip(df['premise'], df['hypothesis'])))
        true_labels.extend([i] * len(df))

    return data, true_labels

# Load model
model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

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

# Combine contradiction and neutral into a single non-entailment column
dataset2_non_entailment = pd.concat([
    dataset2_df[['premise', 'contradiction']].rename(columns={'contradiction': 'hypothesis'}),
    dataset2_df[['premise', 'neutral']].rename(columns={'neutral': 'hypothesis'})
], ignore_index=True)

# Process and evaluate dataset 1 (2 classes)
#dataset1_report = nli(model, dataset1_non_entailment, dataset1_entailment, eval=True)
# Process and fine-tune on metaphor dataset
train_data, train_labels = convert_data(dataset1_non_entailment, dataset1_entailment)
# Make DataLoder object
dataset = NLIDataset([pair[0] for pair in train_data], [pair[1] for pair in train_data], train_labels)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
# Fine-tune model
#new_model = new_train(model, train_dataloader)
#new_model = depricated_train(model, train_dataloader)

# Process and evaluate dataset 2 (2 classes)
#test_data, test_labels = convert_data(dataset2_contradiction, dataset2_entailment, dataset2_neutral)
test_data, test_labels = convert_data(dataset2_non_entailment, dataset2_entailment)

dataset2_report =test(model, test_data, test_labels)
#dataset2_finetuned_report = test(new_model, test_data, test_labels)

#print("Dataset 2 report:")
print(dataset2_report)
