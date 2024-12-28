from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EsmForSequenceClassification, EsmForMaskedLM
#import evaluate
import torch
import numpy as np
import pandas as pd
import wandb
import os
import re
import pdb

from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, tokenizer, input_ids, attention_mask, labels):
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        # Préparer les labels, ici ils sont identiques aux input_ids pour le débruitage
        self.labels = labels

    def shift_tokens_right(self, input_ids, pad_token_id):
        """Décale les tokens vers la droite pour préparer decoder_input_ids."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        shifted_input_ids = input_ids.clone()
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = pad_token_id

        if input_ids.dim() == 1:
            shifted_input_ids = shifted_input_ids.squeeze(0)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Obtenir input_ids et attention_mask pour un index spécifique
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]
        #labels = None

        if input_ids is None : 
            print(f"input_ids {idx} is None")

        for_save = True
        if for_save :
            return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids  # assurez-vous que les labels sont corrects pour votre cas d'utilisation
        }
        # print(input_ids)
        # print(attention_mask)
        # print(labels)

        #if input_ids.dim() == 1:
        #    input_ids = input_ids.unsqueeze(0)
        #if labels.dim() == 1:
        #    labels = labels.unsqueeze(0)

        # Préparer les decoder_input_ids
        decoder_input_ids = input_ids #self.shift_tokens_right(input_ids, self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels  # assurez-vous que les labels sont corrects pour votre cas d'utilisation
        }



# Instantiating and testing out the model
# mod_name = 'esm2_t36_3B_UR50D'
mod_name = 'esm2_t36_3B_UR50D'
model_checkpoint = f'facebook/{mod_name}'

print('Training started for: ', mod_name)

torch.cuda.empty_cache()

# os.environ["WANDB_PROJECT"] = f"vog-classifier-{mod_name}"  # name your W&B project
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_train_epochs = 6
learning_rate = 1e-5

# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project=f"vog-classifier-{mod_name}",
#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": learning_rate,
#         "epochs": num_train_epochs,
#     }
# )

print('Reading data')
# train_dataset = pd.read_csv("/home/thibaut/protrans/train.tsv")
train_dataset = pd.read_csv("/home/rsawhney/PooledAAEmbeddings/tmp/tmp_data_dir/train.tsv")
# Keep only vogs with 10 sequences since we classify back to the vog

vog_min_number_seqs = 10
is_gt_than_min = train_dataset["#GroupName"].value_counts() >= vog_min_number_seqs
vogs_gt_min = train_dataset["#GroupName"].value_counts()[is_gt_than_min].index.unique()
cols_of_interest = ["#GroupName", "ProteinIDs", "sequence"]
dataset = train_dataset[train_dataset["#GroupName"].isin(vogs_gt_min)][cols_of_interest]
dataset = dataset[:100] # TODO: remove


new_ids = dataset.ProteinIDs.str.split(".").apply(lambda x: ".".join(x[1:]))
dataset["ProteinIDs"] = new_ids

dataset.columns = ["labels", "protein_id", "protein_seq"]

#train_df, test_df = train_test_split(dataset, test_size=0.05)
#train_df, test_df = train_test_split(dataset, test_size=0.1, random_state=1)
train_df, test_df = train_test_split(dataset, test_size=0.1, stratify=dataset['labels'], random_state=1)

labels = set(dataset.labels.tolist())
id2label = {i:l for i,l in enumerate(labels)}
label2id = {l:i for i,l in enumerate(labels)}
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = EsmForMaskedLM.from_pretrained(model_checkpoint, num_labels=len(labels), id2label=id2label, label2id=label2id)

print(model)

loraft = True

lora_params = []
if loraft : 

    targets = ["query","key","value"]

    lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            init_lora_weights="gaussian",
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            # modules_to_save = "classifier",
            layers_to_transform = [i for i in range(36) if i % 1 == 0], 
        )

    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
            param.requires_grad = False

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            lora_params.append(param)

pdb.set_trace()

# device_ids =list(range(0, torch.cuda.device_count()))
# print("device_ids", device_ids)

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()


#train_df["protein_seq"] = [" ".join(list(re.sub(r"[UZOB-]", "X", sequence))) for sequence in train_df["protein_seq"]]
#test_df["protein_seq"] = [" ".join(list(re.sub(r"[UZOB-]", "X", sequence))) for sequence in test_df["protein_seq"]]
print(train_df)

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(train_df['protein_seq'], max_length = 1024, add_special_tokens=True, padding="max_length",truncation=True, return_tensors='pt')
test_ids = tokenizer.batch_encode_plus(test_df['protein_seq'], max_length = 1024, add_special_tokens=True, padding="max_length",truncation=True, return_tensors='pt')

print(ids.keys())

input_ids = torch.tensor(ids['input_ids'])
attention_mask = torch.tensor(ids['attention_mask'])
train_df['labels'] = train_df['labels'].astype('category')
train_df['labels_codes'] = train_df['labels'].cat.codes
labels = torch.tensor(train_df['labels_codes'].values, dtype = torch.long)

test_input_ids = torch.tensor(test_ids['input_ids'])
test_attention_mask = torch.tensor(test_ids['attention_mask'])
test_df['labels'] = test_df['labels'].astype('category')
test_df['labels_codes'] = test_df['labels'].cat.codes
test_labels = torch.tensor(test_df['labels_codes'].values, dtype = torch.long)


train_dataset = CustomDataset(tokenizer= tokenizer, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
test_dataset = CustomDataset(tokenizer= tokenizer, input_ids=test_input_ids, attention_mask=test_attention_mask, labels = test_labels)



# test_list = [data for data in test_dataset]
# import pickle
# with open('/home/thibaut/protrans/esm_test_set.pkl', 'wb') as file :
#     pickle.dump(test_list, file)

# print(1+'e')
# generate the DatasetDict
# trn = Dataset.from_dict({'labels':train_df.labels,'protein_id':train_df.protein_id, 'protein_seq': train_df.protein_seq})
# tst = Dataset.from_dict({'labels':test_df.labels,'protein_id':test_df.protein_id, 'protein_seq': test_df.protein_seq})
# dataset_dict = DatasetDict({'train': trn, 'validation': tst})

# accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
# def compute_metrics(p):
#    predictions, labels = p
#    predictions = np.argmax(predictions, axis=1)
#    return accuracy.compute(predictions=predictions, references=labels)


# Tokenize dataset fpr 
def tokenize_and_format(dataset):
    tokenized_outputs = tokenizer.batch_encode_plus(
        dataset['protein_seq'],
        max_length=1022,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    # Ensure labels are in the correct format, e.g., a list of integers
    tokenized_outputs['labels'] = torch.tensor([label2id[x] for x in dataset['labels']])
    return tokenized_outputs

# tokenized_dataset = dataset_dict.map(tokenize_and_format, batched=True, batch_size=50)

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results-esm-mlm",
    evaluation_strategy="epoch",
    do_eval=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    gradient_accumulation_steps=32,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    report_to="wandb",
    logging_steps=10,  # how often to log to W&B
    save_safetensors = False,
    save_steps = 200,
)

# # Initialize the Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Now initialize the Trainer with these datasets
print('Running trainer')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
    #compute_metrics=compute_metrics
)
#ckpt = '/home/thibaut/protrans/esm_3b_cls_1024/results-esm-cls-lora-1024/checkpoint-2000/'
#trainer.train(resume_from_checkpoint=ckpt)
checkpoint = "./results-esm-mlm/checkpoint-800/"
trainer.train(resume_from_checkpoint = checkpoint)
print('Model trained')

out_name = f"vog_{mod_name}_{num_train_epochs}ep_{learning_rate}lr_ns"

trainer.save_model(out_name)
model.base_model.save_pretrained(save_directory = "./results-esm-mlm", safe_serialization = False)
print(f'Saved trainer as {out_name}')

# pdb.set_trace()

# # Train the model
# trainer.train()
# trainer.state
# trainer.state.log_history
# #trainer.train()
# x = trainer.state.log_history
# trainer.state.log_history
# trainer.train()
# trainer.state.log_history
# len(trainer.state.log_histor)
# len(trainer.state.log_history)
# trainer.state.log_history
# x
# x += trainer.state.log_history
# x
# len(x)
# trainer.train()
# x += trainer.state.log_history
# x
# trainer.train()
# x += trainer.state.log_history
# x
# trainer.train()
# x += trainer.state.log_history
# x
# trainer.train()
# x += trainer.state.log_history
# x
# trainer.train()
# history
