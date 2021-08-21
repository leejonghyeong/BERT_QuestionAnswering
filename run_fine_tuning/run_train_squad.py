from torch.nn.utils.clip_grad import clip_grad_norm_
from .squad_utils import find_ends, get_answer_mask, get_index_order_mask, pred_index, get_question_mask
from transformers import BertModel
import numpy as np
from torch.nn.utils import clip_grad_norm


def train_epoch(model, criterion, optimizer, batch_size, answer_indices, bert_inputs, device):

    pred_scores = model(bert_inputs)

    optimizer.zero_grad()
    loss = criterion(pred_scores, answer_indices)
    loss.backward()

    clip_grad_norm_(model.parameters(), max_norm= 1.0)

    optimizer.step()


    
    return loss.item()
