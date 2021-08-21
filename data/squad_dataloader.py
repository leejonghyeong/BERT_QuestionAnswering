import torch
import json
from tqdm.notebook import tqdm

from torch.utils.data import Dataset
from .data_utils import json_to_list
from ..run_fine_tuning.squad_args import squad_args

class SquadDataset(Dataset):
    def __init__(self, infile):
        super().__init__()

        self.args = squad_args()
        self.sentences_ret = []
        self.answers = []
        self.answer_indices = []
        self.sentences = json_to_list(infile)
        
        for i in tqdm(range(len(self.sentences)), desc = f"Loading {infile}"):
            data = json.loads(self.sentences[i])

            question = data["question"]
            context = data["context"]
            answer = data["answer"]
            answer_indices = data["answer_indices"]
            
            self.answers.append(answer)
            self.answer_indices.append(answer_indices)
            self.sentences_ret.append(
                [self.args.cls_token_id] +
                question +
                [self.args.sep_token_id] +
                context +
                [self.args.sep_token_id]
            )
            

    def __len__(self):
        assert len(self.answers) == len(self.sentences_ret)
        return len(self.sentences_ret)


    def __getitem__(self, item):
        #정답 인덱스가 하나씩 밀려있는것 같아서(cls 토큰 미고려로 인해)
        #일단 임시로 인덱스에 1씩 더해주겠음

        return (torch.tensor(self.answers[item]),
                torch.tensor(self.answer_indices[item]) + 1,
                torch.tensor(self.sentences_ret[item]))