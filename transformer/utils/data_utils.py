import os, glob, math, random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from itertools import chain
from datasets import load_dataset

def load_parquet_dataset(train_pattern, valid_pattern, test_pattern):
    train_files = sorted(glob.glob(train_pattern))
    valid_files = sorted(glob.glob(valid_pattern))
    test_files  = sorted(glob.glob(test_pattern))
    data_files = {}
    if train_files: data_files['train'] = train_files
    if valid_files: data_files['validation'] = valid_files
    if test_files:  data_files['test'] = test_files
    if not data_files: raise FileNotFoundError("No parquet files found.")
    return load_dataset("parquet", data_files=data_files)

def build_vocab_from_texts(texts, vocab_size=30000, mask_token="<mask>"):
    words = list(chain.from_iterable([t.split() for t in texts]))
    most_common = Counter(words).most_common(vocab_size - 2)
    stoi = {w: i + 2 for i, (w, _) in enumerate(most_common)}
    stoi['<unk>'] = 0
    stoi[mask_token] = 1
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos

def encode_texts_to_ids(texts, stoi):
    unk = stoi['<unk>']
    return [stoi.get(w, unk) for t in texts for w in t.split()]

class MaskedLMDataset(Dataset):
    def __init__(self, tokens, seq_len=64, mask_prob=0.15, mask_id=1):
        self.tokens = tokens
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.mask_id = mask_id
    def __len__(self):
        return max(0, len(self.tokens) // self.seq_len)
    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = self.tokens[s:s+self.seq_len]
        if len(x) < self.seq_len:
            x += [0] * (self.seq_len - len(x))
        x = torch.tensor(x)
        y = x.clone()
        mask = torch.rand(x.shape) < self.mask_prob
        x[mask] = self.mask_id
        y[~mask] = -100
        return x, y
