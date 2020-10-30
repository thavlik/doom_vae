import os
import numpy as np
import torch
import torch.utils.data as data


class DoomMultiframeDataset(data.Dataset):
    def __init__(self,
                 endpoint: str,
                 bucket: str,
                 num_frames: int,
                 cache=None):
        super(DoomMultiframeDataset, self).__init__()
        self.num_frames = num_frames

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
