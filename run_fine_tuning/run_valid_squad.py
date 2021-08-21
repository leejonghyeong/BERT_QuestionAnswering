import numpy as np
from .squad_utils import (f1_token_score, f1_string_score)
import torch
from transformers import BertTokenizer

def eval_epoch(model, answers, bert_inputs, batch_size, tokenizer: BertTokenizer):

    pred_scores = model(bert_inputs)
    _, pred_indices = pred_scores.max(dim=1)
    
    #predict answer
    pred_answers = []
    for i in range(batch_size):
        start = pred_indices[i][0]
        end = pred_indices[i][1]
        answer = bert_inputs[i][start: end + 1]
        pred_answers.append(answer)


    #F1 score for true answer and pred answer
    f1 = f1_string_score(pred_answers, answers, batch_size, tokenizer)
    
    return f1