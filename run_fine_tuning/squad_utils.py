from typing import Tuple, List
from torch import Tensor
import numpy as np
import collections
import torch
from transformers import BertTokenizer
import string
import re

def normalizer(text: str)-> str:
    def remove_white_space(text: str)->str:
        return "".join(text.split())
    def remove_punctuation(text: str)->str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def remove_article(text: str)->str:
        '''
        관사(article)를 제거한다.
        '''
        pattern = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(pattern, "", text)
        
    return remove_punctuation(remove_white_space(remove_article(text)))

def f1_string_score(pred_answers: Tensor, answers: Tensor, batch_size: int, tokenizer: BertTokenizer):
    f1 = []

    for i in range(batch_size):

        pred_tokens = pred_answers[i].tolist()
        ans_tokens = answers[i].tolist()

        pred = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        ans = tokenizer.decode(ans_tokens, skip_special_tokens=True)

        pred = normalizer(pred)
        ans = normalizer(ans)

        common = collections.Counter(pred) & collections.Counter(ans)
        num_common = sum(common.values())

        if num_common == 0:
            return 0
        
        precision = num_common / len(pred)
        recall = num_common / len(ans)

        score = (2* precision * recall) / (precision + recall)
        f1.append(score)

    return np.mean(f1)


def f1_token_score(pred_answers: Tensor, answers: Tensor, batch_size: int):
    f1 = []

    for i in range(batch_size):
        
        pred = pred_answers[i].tolist()
        ans = answers[i].tolist()

        common = collections.Counter(pred) & collections.Counter(ans)
        num_common = sum(common.values())
        if num_common == 0:
            return 0
        precision = num_common / len(pred)
        recall = num_common / len(ans)

        score = (2 * precision * recall) / (precision + recall)
        f1.append(score)
    
    return np.mean(f1)

def find_ends(inputs):
    ''' 0으로 padding된 answers tensor에서 start token id와 end token id를 반환하는 함수'''
    ends = []
    batch_size = inputs.size(0)
    for i in range(batch_size):
        seq = [x for x in inputs[i] if x!= 0]
        start = seq[0]
        end = seq[-1]
        ends.append([start, end])
    
    return torch.tensor(ends, device = inputs.device)

def pred_index(scores, sep_token_indices, batch_size, n_seq, device):
    '''argmax of T_i \cdot S + T_j \cdot E for j >= i 를 구하는 함수
    
    Args:
        scores:
            permutation of score list which is a list of lists of prediction scores
    
    '''

    question_mask = torch.tensor([[True if j <= sep_token_indices[i] else False for j in range(n_seq)] for i in range(batch_size)], device = device).unsqueeze(2).repeat(1,1,n_seq)
    
    start_score = scores[0].unsqueeze(2).repeat(1,1,n_seq)
    end_score = scores[1].unsqueeze(1).repeat(1,n_seq,1)

    start_score_masked = start_score.triu().masked_fill_(question_mask, 0)
    end_score_masked = end_score.triu().masked_fill_(question_mask, 0)
    
    #(batch_size, 1)
    pred_score_1d = (start_score_masked + end_score_masked).view(batch_size, -1)

    #(batch_size, 1)
    argmax_1d = pred_score_1d.argmax(1).view(-1, 1)
    argmax_2d_i = argmax_1d // n_seq
    argmax_2d_j = argmax_1d % n_seq

    #(batch_size, 2)
    argmax = torch.cat((argmax_2d_i, argmax_2d_j), dim=1)

    return argmax


def get_question_mask(size: Tensor, sep_token_indices: Tensor):
    '''
    Args:
        size (list):
            size of bert_inputs, (batch_size, n_seq)
        sep_token_indices (list):
            list of indices which indicate [SEP] tokens
    '''

    question_mask = []
    for i in range(size[0]):
        index = sep_token_indices[i]
        mask = torch.tensor([True if j <= index else False for j in range(size[1])])
        question_mask.append(mask)
    
    return torch.stack(question_mask)

def get_index_order_mask(size: Tensor, pred_indices: Tensor):
    '''
    If start index is bigger than end index, mask scores corresponding these two indices so that we can give penalty.
    
    Args:
        size (list):
            size of bert_inputs, (batch_size, n_seq)
        pred_indices (list):
            list of start and end indices
    '''

    index_order_mask = []
    for i in range(size[0]):
        index_start = pred_indices[i][0]
        index_end = pred_indices[i][1]

        if index_start <= index_end:
            mask = torch.tensor([False for j in range(size[1])])
        else:
            mask = torch.tensor([True if j == index_start or j == index_end else False for j in range(size[1])])
        
        index_order_mask.append(mask)

    return torch.stack(index_order_mask)


def get_answer_mask(batch_size, n_seq, answer_indices, device):
    
    mask = torch.tensor((), dtype=int, device= device)
    mask = mask.new_zeros([batch_size, n_seq])

    for i in range(batch_size):
        start = answer_indices[i][0]
        end = answer_indices[i][1]
        for j in range(n_seq):
            if j >= start and j < end:
                mask[i][j] = 1

    return mask.to(device)
