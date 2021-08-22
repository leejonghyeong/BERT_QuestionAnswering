import torch
from torch.functional import Tensor
import torch.nn as nn
import numpy as np
from numpy import random

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import notebook
import json

import sentencepiece as spm

'''
fix random seed
& load vocab
'''

np.random.seed(0) 

vocab_infile = ''
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_infile)
mask_token_id = vocab.EncodeAsIds("[MASK]")
cls_token_id = vocab.EncodeAsIds("[CLS]")
sep_token_id = vocab.EncodeAsIds("[SEP]")


''' config:  n_head, d_head, n_layer '''

def encoder_mask(inputs):
    n_seq = inputs.size(1)

    self_mask = inputs.unsqueeze(1).repeat(1, n_seq, 1).contiguous()
    zero_mask = torch.zeros_like(self_mask)
    enc_mask = torch.eq(self_mask, zero_mask)

    return enc_mask


def sinusoid_encoding_table(n_seq, d_hidn):
    def pos_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def pos_vector(position):
        return [pos_angle(position, i_hidn) for i_hidn in range(d_hidn)]
    
    sinusoidtable = [pos_vector(position) for position in range(n_seq)]
    sinusoidtable[:, 0::2] = np.sin(sinusoidtable[:, 0::2])
    sinusoidtable[:, 1::2] = np.cos(sinusoidtable[:, 1::2])

    return sinusoidtable

class SegmentEmbedding(nn.Module):
    def __init__(self, config):
        
        self.seg_emb1 = nn.Embedding(self.config.n_max_seq, self.config.d_hidn)
        self.seg_emb2 = nn.Embedding(self.config.n_max_seq, self.config.d_hidn)

    def forward(self, inputs, sep_token_indices):
        '''inputs 문장 하나씩 가져와서 seg_emb1, seg_emb2 적용하고 concatenate'''
        seg_emb_rtn = []
        for i, sentence in enumerate(inputs):
            index = sep_token_indices[i]
            sentence = torch.cat((self.seg_emb1(sentence[:index+1]), self.seg_emb2(sentence[index+1:])), 0)
            seg_emb_rtn.append(sentence)
        
        return torch.stack(seg_emb_rtn)


class ScaledDotProduct(nn.Module):
    def __init__(self, d_head):
        self.scale = 1 / (d_head ** 0.5)

    def forward(self, Q,K,V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1,-2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        
        attn_prob = nn.Softmax(dim = -1)(scores)
        context = torch.matmul(attn_prob, V)

        return context

class PostiveFeedForward(nn.Module):
    def __init__(self, d_hidn):
        self.d_hidn = d_hidn
        self.linear1 = nn.Linear(d_hidn, 4 * d_hidn)
        self.linear2 = nn.Linear(4 * d_hidn, d_hidn)

    def forward(self, inputs):
        outputs = self.linear1(inputs)
        outputs = self.linear2(outputs)

        return outputs


class EncoderLayer(nn.Module):
    def __init__(self, config):
        self.d_head = config.d_head
        self.n_head = config.n_head
        self.d_hidn = config.d_hidn

        self.w_q = nn.Linear(self.d_hidn, self.d_hidn)
        self.w_k = nn.Linear(self.d_hidn, self.d_hidn)
        self.w_v = nn.Linear(self.d_hidn, self.d_hidn)

        self.normalization1 = nn.LayerNorm(self.d_hidn)
        self.normalization2 = nn.LayerNorm(self.d_hidn)
        self.pos_ffn = PostiveFeedForward(self.d_hidn)

        self.scaled_dot_prod = ScaledDotProduct(self.d_head)
    
    def forward(self, inputs, attn_mask):
        Q = inputs
        K = inputs
        V = inputs
        '''MultiHeadAttention'''
        batch_size = Q.size(0)
        n_seq = Q.size(1)

        q_s = self.w_q(Q).view(batch_size, n_seq, self.n_head, self.d_head).transpose(1,2)
        k_s = self.w_k(K).view(batch_size, n_seq, self.n_head, self.d_head).transpose(1,2)
        v_s = self.w_v(V).view(batch_size, n_seq, self.n_head, self.d_head).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        attn_outputs = self.scaled_dot_prod(q_s, k_s, v_s, attn_mask).transpose(1,2)
        attn_outputs = attn_outputs.contiguous().view(batch_size, n_seq, self.d_hidn)

        '''Add and Normalization'''
        pos_ffn_inputs = self.normalization1(attn_outputs + inputs)

        '''Positive FeedForward'''
        pos_ffn_outputs = self.pos_ffn(pos_ffn_inputs)

        '''Add and Normalization2'''
        enc_outputs = self.normalization2(pos_ffn_outputs + pos_ffn_inputs)

        return enc_outputs 

    

class BERT(nn.Module):
    def __init__(self, config):
        self.config = config

        '''Token embedding
            수정: 맨 앞에 [CLS] token 넣기'''
        self.enc_emb = nn.Embedding(self.config.n_vocab, self.config.d_hidn)

        '''positional embedding'''
        sinusoid = torch.FloatTensor(sinusoid_encoding_table
        (self.config.n_max_seq + 1, self.config.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid, freeze=True)

        '''segment embedding'''
        self.seg_emb = SegmentEmbedding(self.config)
        
        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
        
    
    def forward(self, inputs, sep_token_indices)->Tensor:
        batch_size = inputs.size(0)
        n_seq = inputs.size(1)
        
        positions = torch.arange(n_seq, device = inputs.device, dtype = inputs.dtype).expand(batch_size, n_seq).contiguous() + 1
        attn_mask = encoder_mask(n_seq)

        outputs = self.enc_emb(inputs) + self.pos_emb(positions) + self.seg_emb(inputs, sep_token_indices)

        attn_probs = []

        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)

        return outputs, self.enc_emb


#Data Loader and Masking

def masking_input_tokens(input, probability):
    input_length = len(input)
    input_indices = range(input_length)
    avg_mask_num = round(probability * input_length)

    if avg_mask_num == 0:
        if random.rand() <= 0.15:
            avg_mask_num = 1

    rand_indices = random.choice(input_indices, avg_mask_num)
    
    for index in rand_indices:
        rand_num_token = random.rand()

        if rand_num_token > 0.2:
            input[index] = mask_token_id
        elif rand_num_token <= 0.2 and rand_num_token > 0.1:
            input[index] = random.randint(30007)

    return input


class BERTDataset(torch.utils.data.Dataset):
    '''data 로드하고 [cls], [sep] 토큰 집어넣기 '''
    '''single sentence인지 task specific pair of sentences인지 '''
    def __init__(self, vocab, infile, task):
        self.task = task
        self.sentences = list(infile.readlines())
        self.sentences_rtn = []
        self.masked_sentences_rtn = []
        self.sep_token_indices = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(notebook.tqdm(f, total = line_cnt, desc = 
                                                    f"Loading {infile}", unit =" lines")):
                data = json.loads(line)

                if self.task == 'Pretrain':
                    # [CLS] sentence A [SEP] sentence B [SEP]
                    # 50 % 확률로 A, B는 연속된 문장
                    # 연속되면 'IsNext' 아니면 'NotNext' 라벨에 추가

                    self.labels = []
                    sentenceA = data["text"]
                    self.sep_token_indices.append(len(sentenceA) + 1)

                    if random.random_sample >= 0.5:
                        random_data = json.loads(random.choice(self.sentences))
                        sentenceB = random_data["text"]
                        self.labels.append(vocab.EncodeAsIds('NotNext'))
                    else:
                        next_data = json.loads(self.sentences[i + 1])
                        sentenceB = next_data["text"]
                        self.labels.append(vocab.EncodeAsIds('IsNext'))

                    self.sentences_rtn.append(
                        cls_token_id +
                        sentenceA +
                        sep_token_id +
                        sentenceB +
                        sep_token_id
                    )

                    self.masked_sentences_rtn.append(
                        cls_token_id +
                        masking_input_tokens(sentenceA) +
                        sep_token_id +
                        masking_input_tokens(sentenceB) +
                        sep_token_id
                    )
                
                elif self.task == 'SQuAD':
                    #[CLS] question [sep] context [sep]
                    self.answers = []
                    question = data["question"]
                    context = data["context"]
                    self.answers.append(data["answer"])
                    self.sep_token_indices.append(len(question) + 1)

                    self.sentences_rtn.append(
                        cls_token_id +
                        question +
                        sep_token_id +
                        context +
                        sep_token_id
                    )
        
    def __len__(self):
        if self.task == "Pretrain":
            assert len(self.labels) == len(self.sentences_rtn)
        elif self.task == "SQuAD":
            assert len(self.answers) == len(self.sentences_rtn)

        return len(self.sentences_rtn)


    def __getitem__(self, item):
        if self.task == "Pretrain":
            return (torch.tensor(self.sep_token_indices[item]),
                    torch.tensor(self.labels[item]),
                    torch.tensor(self.sentences_rtn[item]),
                    torch.tensor(self.masked_sentences_rtn[item]))
        elif self.task == "SQuAD":
            return (torch.tensor(self.sep_token_indices[item]),
                    torch.tensor(self.answers[item]),
                    torch.tensor(self.sentences_rtn[item]))


def pretrain_collate_fn(inputs):
    '''배치단위로 처리, padding'''
    sep_token_indices, labels, sentences, masked_sentences = list(zip(*inputs))

    bert_inputs = torch.nn.utils.rnn.pad_sequence(sentences, batch_first= True, padding_value= 0)
    masked_bert_inputs = torch.nn.utils.rnn.pad_sequence(masked_sentences, batch_first=True, padding_value= 0)

    batch = [torch.stack(sep_token_indices), torch.stack(labels), bert_inputs, masked_bert_inputs]


def SQuAD_collate_fn(inputs):
    sep_token_indices, answers, sentences = list(zip(*inputs))

    bert_inputs = torch.nn.utils.rnn.pad_sequence(sentences, batch_first = True, padding_value = 0)

    batch = [torch.stack(sep_token_indices), torch.stack(answers), bert_inputs]


#Pretrain BERT

def first_token_is_label(inputs, labels):
    L = inputs.size(1)
    indices = torch.arange(L+1)[torch.arange(L+1) != 1]
    labels_reshaped = labels.unsqueeze(0).transpose(0,1)
    concat = torch.cat((labels_reshaped, inputs))

    return torch.index_select(concat, 1, indices)

def pretrain_epoch(config, model: BERT, criterion, optimizer, data_loader):
    losses = []
    model.train()

    for i, value in enumerate(data_loader):
        sep_token_indices, labels, bert_inputs, masked_bert_inputs = map(lambda x: x.to(config.device), value)

        #(batch_size, n_seq)
        bert_inputs = first_token_is_label(bert_inputs, labels)
        bert_outputs, token_emb = model(masked_bert_inputs, sep_token_indices)

        #(batch_size, n_seq, n_vocab)
        pretrain_scores = torch.matmul(bert_outputs, token_emb.transpose(0,1))

        optimizer.zero_grad()

        loss = criterion(pretrain_scores, bert_inputs)
        losses.append(loss.item())
        loss.backward()

        optimizer.step()
    
    return np.mean(losses)

def pretrain_BERT(config, vocab):

    model = BERT(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters, lr= config.learning_rate)

    train_dataset = BERTDataset(vocab, './bookcorpus_train.json','Pretrain')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True, collate_fn = pretrain_collate_fn)
    
    losses = []

    for epoch in range(config.n_epoch):
        loss = pretrain_epoch(config, model, criterion, optimizer, train_loader)
        losses.append(loss)
