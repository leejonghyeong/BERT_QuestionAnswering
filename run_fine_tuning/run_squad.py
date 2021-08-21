import torch
import torch.nn as nn
import numpy as np
from numpy import random

import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from data.squad_dataloader import SquadDataset
from data.data_utils import squad_collate_fn
from .run_train_squad import train_epoch
from .run_valid_squad import eval_epoch
from .squad_args import squad_args
from transformers import BertModel

random.seed(0)
torch.random.manual_seed(0)

#configuration
args = squad_args()
tokenizer = args.tokenizer
batch_size = args.batch_size
learning_rate = args.learning_rate
n_epoch = args.n_epoch
device = args.device
d_hidden = args.d_hidden

#SQuAD train and eval
class QuestionAnswering(nn.Module):
    def __init__(self, device, d_hidden):
        super().__init__()

        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.projection = nn.Linear(d_hidden, 2)

    def get_token_emb(self):
        return self.model.get_input_embeddings()

    def forward(self, inputs):
        bert_outputs = self.model(inputs)[0]
        pred_scores = self.projection(bert_outputs)

        return pred_scores

model = QuestionAnswering(device, d_hidden)
model.to(device)
token_emb = model.get_token_emb()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

train_dataset = SquadDataset('./SQuAD_mini.json')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = squad_collate_fn, drop_last= True)

valid_dataset = SquadDataset('./SQuAD_mini.json')
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, collate_fn = squad_collate_fn, drop_last = True)

losses = []
scores = []

for epoch in range(n_epoch):
    #train_epoch
    model.train()

    with tqdm(total = len(train_loader), desc = f"Train {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            sep_token_indices, _, answer_indices, bert_inputs = map(lambda x:x.to(device), value)

            loss = train_epoch(model, criterion, optimizer, batch_size, answer_indices, bert_inputs, device)
            losses.append(loss)

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss:.3f} ({np.mean(losses):.3f})")


    #valid_epoch
    model.eval()

    with tqdm(total = len(valid_loader), desc = f"Valid {epoch}") as pbar:
        for i, value in enumerate(valid_loader): 
            sep_token_indices, answers, _, bert_inputs = map(lambda x:x.to(device), value)

            score = eval_epoch(model, token_emb, answers, sep_token_indices, bert_inputs, batch_size, device, tokenizer)
            scores.append(score)

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {score:.3f}")


