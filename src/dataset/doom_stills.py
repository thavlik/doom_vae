import os
import numpy as np
import torch
import torch.utils.data as data


class DoomStillsDataset(data.Dataset):
    def __init__(self,
                 endpoint: str,
                 bucket: str,
                 cache=None):
        super(DoomStillsDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
