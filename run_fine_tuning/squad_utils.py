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
