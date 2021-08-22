from torch import Tensor
import numpy as np
import collections
from transformers import BertTokenizer
import string
import re

def normalizer(text: str)-> str:
    '''
    공백, 문장부호, 관사를 차례로 제거 
    '''
    def remove_white_space(text: str)->str:
        return "".join(text.split())
    def remove_punctuation(text: str)->str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def remove_article(text: str)->str:
        pattern = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(pattern, "", text)
        
    return remove_punctuation(remove_white_space(remove_article(text)))

def f1_string_score(
    pred_answers: Tensor, 
    answers: Tensor, 
    batch_size: int, 
    tokenizer: BertTokenizer
    )->float:
    '''
    Compute f1 score in string level

    Args:
        pred_answers (:obj:`Tensor`):
            a tensor of encoded answers predicted by Bert QuestionAnswering Model
        answers (:obj:`Tensor`):
            a tensor of encoded ground answers
        batch_size (:obj:`int`)

        tokenizer (:obj:`BertTokenizer`)
    '''
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


def f1_token_score(
    pred_answers: Tensor, 
    answers: Tensor, 
    batch_size: int
    )->float:
    '''
    Compute f1 score in token level

    Args:
        pred_answers (:obj:`Tensor`):
            a tensor of encoded answers predicted by Bert QuestionAnswering Model
        answers (:obj:`Tensor`):
            a tensor of encoded ground answers
        batch_size (:obj:`int`)
    '''
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
