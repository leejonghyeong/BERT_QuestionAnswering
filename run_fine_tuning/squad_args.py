from transformers import BertTokenizer
import torch

class squad_args():
    def __init__(self, **kwargs) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.batch_size = kwargs.pop('batch_size', 4)
        self.d_hidden = kwargs.pop('d_hidden', 768)
        self.max_length = kwargs.pop('max_length', 512)
        self.learning_rate = kwargs.pop('learning_rate', 5e-5)
        self.n_epoch = kwargs.pop('n_epoch', 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
