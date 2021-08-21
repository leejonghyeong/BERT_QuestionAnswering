import torch
import collections
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer
import re
import string

def get_answer_mask(batch_size, n_seq, answer_indices):
    
    mask = torch.tensor((), dtype= int)
    mask = mask.new_zeros([batch_size, n_seq])

    for i in range(batch_size):
        start = answer_indices[i][0]
        end = answer_indices[i][1]
        for j in range(n_seq):
            if j >= start and j < end:
                mask[i][j] = 1

    return mask

x = torch.FloatTensor([
    [2,3],
    [5,1]
])

A = torch.tensor([
    [0,1],[2,3],[2,4],[3,5]
])

print(A.size(1))

mask = get_answer_mask(2,4,A)
print(mask)

#mask is FloatTensor
