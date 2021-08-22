from torch.nn.utils.clip_grad import clip_grad_norm_
from transformers import BertModel
from torch import Tensor


def train_epoch(
    model: BertModel, 
    criterion, 
    optimizer, 
    answer_indices: Tensor, 
    bert_inputs: Tensor
    )->float:
    
    pred_scores = model(bert_inputs)

    optimizer.zero_grad()
    loss = criterion(pred_scores, answer_indices)
    loss.backward()

    clip_grad_norm_(model.parameters(), max_norm= 1.0)

    optimizer.step()

    return loss.item()
