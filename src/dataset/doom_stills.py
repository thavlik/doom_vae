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

if __name__ == '__main__':
    import youtube_dl
    ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s.%(ext)s'})
    with ydl:
        result = ydl.extract_info(
            'http://www.youtube.com/watch?v=BaW_jenozKc',
            download=False # We just want to extract the info
        )
    if 'entries' in result:
        # Can be a playlist or a list of videos
        video = result['entries'][0]
    else:
        # Just a video
        video = result
    print(video)
