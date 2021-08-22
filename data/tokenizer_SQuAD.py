from datasets import load_dataset
from transformers import BertTokenizer
from data.data_utils import tokenizer_squad

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

infile1 = load_dataset('squad', split='train')
outfile1 = './SQuAD_train_final1.json'

infile2 = load_dataset('squad', split='validation')
outfile2 = "./SQuAD_test_final1.json"

tokenizer_squad(infile1, outfile1, tokenizer)
tokenizer_squad(infile2, outfile2, tokenizer)