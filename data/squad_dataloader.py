import torch
import json
from typing import Tuple, List, Optional

from torch import Tensor
from tqdm.notebook import tqdm

from torch.utils.data import Dataset
from .data_utils import json_to_list
from ..run_fine_tuning.squad_args import squad_args

class SquadDataset(Dataset):
    '''
    Construct a Dataset for SQuAD by combining each answer and context pair.

    Args:
        infile (:obj:`str`):
            infile relative path
    '''
    def __init__(self, infile):
        super().__init__()

        self.args = squad_args()
        self.sentence_pairs = []
        self.answers = []
        self.answer_indices = []
        self.contexts = json_to_list(infile)
        
        for i in tqdm(range(len(self.contexts)), desc = f"Loading {infile}"):
            data = json.loads(self.contexts[i])

            question = data["question"]
            context = data["context"]
            answer = data["answer"]
            answer_indices = data["answer_indices"]
            
            self.answers.append(answer)
            self.answer_indices.append(answer_indices)
            self.sentence_pairs.append(
                [self.args.cls_token_id] +
                question +
                [self.args.sep_token_id] +
                context +
                [self.args.sep_token_id]
            )
            

    def __len__(self):
        assert len(self.answers) == len(self.sentence_pairs)
        return len(self.sentence_pairs)


    def __getitem__(self, item)->Tuple(Tensor):

        return (torch.tensor(self.answers[item]),
                torch.tensor(self.answer_indices[item]),
                torch.tensor(self.sentence_pairs[item]))