import torch
from torch._C import device
import torch.nn as nn
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import notebook

from BERT_model import *
from BERT_datasets import *


#Pretrain BERT
#n_epoch 40->10
#batch_size 256->64
config = Config({"n_vocab": 30009
                ,"n_max_seq": 512
                ,"d_hidn": 768
                ,"n_layer": 12
                ,"n_head": 12
                ,"d_head": 64
                ,"batch_size": 256
                ,"n_epoch": 40
                ,"learning_rate" : 1e-4
                ,"L2_penalty": 0.01
                ,"dropout": 0.1})

def first_token_is_label(inputs, labels):
    sequence = inputs.transpose(0,1)
    sequence[0] = labels.squeeze(1)

    return sequence.transpose(0,1)


def pretrain_epoch(epoch,config, model: BERT, criterion, optimizer, data_loader):
    losses = []
    model.train()

    with notebook.tqdm(total = len(data_loader), desc = f"Train {epoch}") as pbar:
        for i, value in enumerate(data_loader):
            sep_token_indices, labels, bert_inputs, masked_bert_inputs = map(lambda x: x.to(config.device), value)

            #(batch_size, n_seq)
            bert_inputs = first_token_is_label(bert_inputs, labels)
            #(batch_size, n_seq, d_hidn), (n_vocab, d_hidn)
            bert_outputs, token_emb = model(masked_bert_inputs, sep_token_indices)

            #copy embedding to linear layer
            token_linear = nn.Linear(config.d_hidn, config.n_vocab)
            token_linear.weight = nn.Parameter(token_emb.weight, requires_grad= True)

            #(batch_size, n_seq, n_vocab)
            pretrain_scores = token_linear(bert_outputs)

            optimizer.zero_grad()

            loss = criterion(pretrain_scores.transpose(1,2), bert_inputs)
            losses.append(loss.item())
            loss.backward()

            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")
    
    return np.mean(losses)

def pretrain_BERT(config):

    model = BERT(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= config.learning_rate, weight_decay= config.L2_penalty)

    train_dataset = BERTDataset('./bookcorpus_train_example2.json','Pretrain')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True, collate_fn = pretrain_collate_fn)
    
    #losses = []

    #for epoch in range(config.n_epoch):
    loss = pretrain_epoch(1, config, model, criterion, optimizer, train_loader)
    #    losses.append(loss)

    #torch.save(model.state_dict(), 'BERT_pretrained.pt')
#model saved 필요
'''
for i in range(100):
    print(vocab.IdToPiece(i))
print(vocab.PieceToId("[MASK]"))
print(vocab.PieceToId("[IsNext]"))
print(vocab.PieceToId("[NotNext]"))
#pretrain_BERT(config)
'''

'''
inputs = torch.tensor([[1,2,3,4,5],[-1,-2,-3,-4,-5]])
labels = torch.tensor([[1],[0]])

L = inputs.size(1)
indices = torch.arange(L+1)[torch.arange(L+1) != 1]
labels_reshaped = labels.unsqueeze(0).transpose(0,1)
print(labels_reshaped)
print(labels)
print(inputs)
concat = torch.cat((labels, inputs), dim =1)

print(concat)
print(torch.index_select(concat, 1, indices))
'''

pretrain_BERT(config)