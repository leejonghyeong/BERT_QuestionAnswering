from typing import List, Optional, Tuple
from numpy import random
from torch import Tensor
import json
import torch

from transformers.models.bert.tokenization_bert import BertTokenizer

def masking_input_tokens(input: Tensor, probability: int, mask_token_id: int, n_vocab: int)-> Tensor:
    '''
    Masking input sentence with given probability. We will mask tokens by [MASK] token for 80%, random token for 10%, and original token for 10%.

    Args:
        input:
            input sentence, list of token ids.
        probablity:
            probability of masking tokens.
        mask_token_id:
            token id of [MASK].
        n_vocab:
            size of vocabulary.
    '''
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
            input[index] = random.randint(n_vocab)

    return input


def random_sentence(sentences: list, i: int, n_lines: int, NotNext_token_id: int, IsNext_token_id: int)->Tuple[list, list]:
    '''
    Return random sentence for 50% prob and next sentence for else

    Args:
        setences (list):
            list of sentences of infile
        i (int):
            index of current sentence
        n_lines (int):
            number of lines in the infile
        NotNext_token_id (int):
            token id of [NotNext]
        IsNext_token_id (int):
            token id of [IsNext]
    '''
    if random.random_sample() >= 0.5 or i == n_lines - 1:
        random_data = json.loads(sentences[random.randint(n_lines)])
        next_sentence = random_data["text"]
        label = [NotNext_token_id]
    else:
        next_data = json.loads(sentences[i + 1])
        next_sentence = next_data["text"]
        label = [IsNext_token_id]

    return next_sentence, label


def find_answer_indices(answer, context, sep_token_index)-> List:
    for i in range(len(context)):
        if answer == context[i: i + len(answer)]:
            return [i + sep_token_index, i + sep_token_index + len(answer) - 1]
        

def squad_collate_fn(inputs):
    sep_token_indices, answers, answer_indices, bert_inputs = list(zip(*inputs))

    answers = torch.nn.utils.rnn.pad_sequence(answers, batch_first = True, padding_value= 0)
    bert_inputs = torch.nn.utils.rnn.pad_sequence(bert_inputs, batch_first = True, padding_value = 0)

    return [torch.stack(sep_token_indices), answers, torch.stack(answer_indices), bert_inputs]


def len_limit_squad(instance: dict, max_length: Optional[int] = 512) -> bool:
    if len(instance["question"]) + len(instance["context"]) <= max_length - 3:
        return False
    
    return True

def tokenizer_squad(infile, outfile, tokenizer: BertTokenizer, max_length: Optional[int] = 512):
    '''
    tokenize SQuAD dataset if given sentence length is less than max_length

    Args:
        infile:
            SQuAD dataset loaded from huggingface
        outfile: 
            json file
        tokenizer:
            BertTokenizer, from huggingface, pretrained for 'bert-base-uncased'
        max_length (int):
            max_length for input sentence length including [CLS], [SEP] tokens

    '''

    with open(outfile, "wt") as f:
        for sentence in infile:
            
            answer = sentence["answers"]["text"][0]
            context = sentence["context"]
            question = sentence["question"]

            ans_id = tokenizer.encode(answer, add_special_tokens= False)
            cont_id = tokenizer.encode(context, add_special_tokens= False)
            quest_id = tokenizer.encode(question, add_special_tokens= False)
            
            #special token 제외했으므로 cls 토큰도 고려해줘야함
            #나중 수정사항: len(quest_id) + 2
            sep_token_id = len(quest_id) + 1
            ans_indices = find_answer_indices(ans_id, cont_id, sep_token_id)

            instance = { 
                "answer": ans_id
                ,"context": cont_id
                ,"question": quest_id
                ,"answer_indices": ans_indices
                }

            if len_limit_squad(instance):
                continue

            if ans_indices == None:
                continue

            f.write(json.dumps(instance)+"\n")

def json_to_list(infile):
    with open(infile, "rt") as f:
        return f.readlines()