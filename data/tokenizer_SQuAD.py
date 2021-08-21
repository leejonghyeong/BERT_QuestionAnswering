import json
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from data.data_utils import tokenizer_squad
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

infile2 = load_dataset('squad', split='train')
outfile2 = './SQuAD_train_final.json'

infile3 = load_dataset('squad', split='validation')
outfile3 = "./SQuAD_test_final.json"

if __name__ == '__main__':
    tokenizer_squad(infile2, outfile2, tokenizer)
    tokenizer_squad(infile3, outfile3, tokenizer)


'''
with open(outfile2, "rt") as f:
    for line in f:
        instance = json.loads(line)
        answer = instance['answer']
        context = instance['context']
        print(tokenizer.batch_decode(answer))
        print(tokenizer.batch_decode(context))
'''