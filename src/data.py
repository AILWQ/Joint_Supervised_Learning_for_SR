import json
from pathlib import Path

import torch
from numpy import *
from torch.utils.data import Dataset

from src.utils import load_metadata_hdf5


class JCLDataset(Dataset):
    def __init__(self, data, metadata_path: Path, cfg, mode: str):

        print("{} dataset after filter has {} examples".format(mode, len(data)))
        metadata = load_metadata_hdf5(metadata_path)
        self.len = metadata.total_number_of_eqs
        self.eqs_per_hdf = metadata.eqs_per_hdf
        self.word2id = metadata.word2id
        self.data_path = metadata_path
        self.mode = mode
        self.cfg = cfg
        self.num_Vars = cfg.dataset_train.num_vars
        self.num_Ys = cfg.dataset_train.num_y
        self.data = data
        self.block_size = self.cfg.dataset_train.block_size
        self.threshold = [-1000, 1000]
        self.number_of_points = cfg.dataset_train.number_of_points

        # build skeleton dictionary
        self.eq_set = {str(json.loads(chunk)['traversal']) for chunk in self.data}
        self.eq2id = {eq: i for i, eq in enumerate(self.eq_set)}

    def __getitem__(self, index):

        chunk = self.data[index]
        try:
            chunk = json.loads(chunk)
        except:
            # try the previous example
            index = index - 1
            index = index if index >= 0 else 0
            chunk = self.data[index]
            chunk = json.loads(chunk)

        # if self.mode == "train" or self.mode == "val":
        traversal = chunk['traversal']
        eq_id = torch.tensor(self.eq2id[str(traversal)], dtype=torch.long)
        # print(traversal)
        # print(chunk['eq'])
        # print(chunk['skeleton'])
        tokenized_expr = tokenize(traversal, self.word2id)
        Padding_size = max(self.block_size - len(tokenized_expr), 0)
        trg = tokenized_expr + [self.cfg.architecture.trg_pad_idx] * Padding_size
        points = torch.zeros(self.num_Vars + self.num_Ys, self.number_of_points)
        for idx, xy in enumerate(zip(chunk['X'], chunk['y'])):
            x = xy[0]  # list x
            # x = [(e-minX[eID])/(maxX[eID]-minX[eID]+eps) for eID, e in enumerate(x)] # normalize x
            x = x + [0] * (max(self.num_Vars - len(x), 0))  # padding

            y = [xy[1]] if type(xy[1]) == float or type(xy[1]) == np.float64 else xy[1]  # list y

            # y = [(e-minY)/(maxY-minY+eps) for e in y]
            y = y + [0] * (max(self.num_Ys - len(y), 0))  # padding
            p = x + y
            p = torch.tensor(p)

            # p = torch.nan_to_num(p, nan=self.threshold[1],
            #                      posinf=self.threshold[1],
            #                      neginf=self.threshold[0])
            points[:, idx] = p

        # points = torch.nan_to_num(points, nan=self.threshold[1],
        #                           posinf=self.threshold[1],
        #                           neginf=self.threshold[0])
        trg = torch.tensor(trg, dtype=torch.long)
        return points, trg, eq_id

    def __len__(self):
        return len(self.data)


def tokenize(prefix_expr: list, word2id: dict) -> list:
    tokenized_expr = []
    tokenized_expr.append(word2id["S"])
    for i in prefix_expr:
        tokenized_expr.append(word2id[i])
    tokenized_expr.append(word2id["F"])
    return tokenized_expr
