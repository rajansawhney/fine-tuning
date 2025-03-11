import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn
import transformers
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from transformers import Trainer, DataCollatorForLanguageModeling, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration, T5ForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel
from torch.distributed import barrier
from Bio import SeqIO
import inspect
import re
import pandas as pd
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

import pdb


# directories
data_dir = "/data/rajan/vog"
pkl_dir = f"{data_dir}/data-pkls"
msa_fasta_dir = f"{data_dir}/fasta"
saved_mod_dir = "saved_mods"

emb_dir = f"{data_dir}/emb"

layer_protTrans_xl = 24

# Single GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('Using gpu:', device)

torch.cuda.empty_cache()

# read data
print('Reading data')

# # NOTE: for seqs
dataset = pd.read_pickle(open("/data/rajan/vog/data-pkls/vog_seq_test_df.pkl", "rb"))

vog_seq_df = dataset.groupby('labels').filter(lambda x: len(x) >= 10)
print(vog_seq_df)
# ## NOTE: regen embs for seq len > 500 # TODO: remove for next time
# vog_seq_df = vog_seq_df[vog_seq_df.protein_seq.str.len()>=500]
# print(vog_seq_df)
pdb.set_trace()

seq_ids = vog_seq_df.protein_id.to_list()
sequences = vog_seq_df.protein_seq.to_list()
sequences = [" ".join(list(re.sub(r"[UZOB-]", "X", sequence))) for sequence in sequences]
labels = set(vog_seq_df.labels.tolist())
id2label = {i:l for i,l in enumerate(labels)}
label2id = {l:i for i,l in enumerate(labels)}


# # NOTE: for MSA
# dataset = pd.read_pickle(open("/data/rajan/vog/data-pkls/new_sel_msa_w_cons_col_df.pkl", "rb"))
# print(dataset)
# seq_ids = dataset.seq_id.to_list()
# sequences = dataset.seq_str.to_list()
# sequences = [" ".join(list(re.sub(r"[UZOB-]", "X", sequence))) for sequence in sequences]
# labels = set(dataset.align_id.tolist())
# id2label = {i:l for i,l in enumerate(labels)}
# label2id = {l:i for i,l in enumerate(labels)}


print('loading mods')
chk_pt = "./checkpoint-27750"

tokenizer = T5Tokenizer.from_pretrained(chk_pt)
base_model = T5ForSequenceClassification.from_pretrained(chk_pt, num_labels=len(labels), id2label=id2label, label2id=label2id)
model = PeftModel.from_pretrained(base_model, chk_pt)
model = model.to(device)
print(model)
# pdb.set_trace()

# # NOTE: for seqs
vog_seq_df['labels'] = vog_seq_df['labels'].astype('category')
vog_seq_df['labels_codes'] = vog_seq_df['labels'].cat.codes
labels = torch.tensor(vog_seq_df['labels_codes'].values, dtype = torch.long).tolist()
output_dir = "/data/rajan/vog/emb/prot_t5_xl/seq/ft_cls_lora"

# # NOTE: for msa
# dataset['align_id'] = dataset['align_id'].astype('category')
# dataset['labels_codes'] = dataset['align_id'].cat.codes
# labels = torch.tensor(dataset['labels_codes'].values, dtype = torch.long).tolist()
# output_dir = "/data/rajan/vog/emb/prot_t5_xl/msa/ft_cls_lora"

# pdb.set_trace()
error_seqs = []
print('Generating embeddings...')

id_seq_zip = zip(seq_ids, sequences)
for i, (seq_id, sequence) in tqdm(enumerate(id_seq_zip), total=len(seq_ids)):
    encoded_tokens = tokenizer.batch_encode_plus([sequence], max_length=1024, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
    encoded_tokens["decoder_input_ids"] = encoded_tokens["input_ids"]
    encoded_tokens["labels"] = torch.tensor(labels[i])
    encoded_tokens = encoded_tokens.to(device)
    try:
        with torch.no_grad():
            outputs = model(**encoded_tokens, output_hidden_states=True)
        embedding = outputs.encoder_last_hidden_state
        torch.save(embedding[0], f'{output_dir}/{seq_id}.pt')
    except Exception as e:
        print(e)
        print('skipping seq_id:', seq_id)
        error_seqs.append(seq_id)
        continue
    
print('embeddings generated and saved')

    
with open('error_seqs.txt', 'w') as f:
    for seq in error_seqs:
        f.write(f"{seq}\n")   

print('Done')