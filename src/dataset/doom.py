import os
import numpy as np
import torch
import torch.utils.data as data
from math import floor
import youtube_dl
import cv2
from torchvision.transforms import Resize


def load_links(path):
    links = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue
            links.append(line)
    return links


def get_raw_frames(id,
                   start_frame: int,
                   num_frames: int):
    # TODO: check if video 
    with youtube_dl.YoutubeDL({'outtmpl': '%(id)s.%(ext)s'}) as ydl:
        video = ydl.extract_info(
            f'https://www.youtube.com/watch?v={id}',
            download=True,
        )
        assert 'entries' not in video, "playlist not supported"
        cap = cv2.VideoCapture(f'{video["id"]}.{video["ext"]}')
        if not cap.isOpened():
            raise ValueError(f"Error opening video stream or file")
        frame_no = 0
        end_frame = start_frame + num_frames
        frames = []
        while(cap.isOpened() and frame_no < end_frame):
            ret, frame = cap.read()
            if ret == True:
                if frame_no >= start_frame:
                    frames.append(frame)
                frame_no += 1
            else:
                break
        cap.release()
        assert len(
            frames) == num_frames, f"expect {num_frames} frames, got {len(frames)}"
        return frames


def resize_frames(frames,
                  width: int,
                  height: int):
    return Resize((height, width))(frames)


def get_frames(id,
               start_frame: int,
               num_frames: int,
               width: int,
               height: int):
    raw_frames = get_raw_frames(id, start_frame, num_frames)
    return resize_frames(raw_frames, width, height)


class DoomDataset(data.Dataset):
    """
    Arguments:
        path (str): path to links.txt

        cache_path (str): video download cache path

        num_frames (int): the number of sequential frames to
        include in each training example

        width (int): output resolution X

        height (int): output resolution Y

        fps (int): target frames per second. If the source video
        FPS differs, the nearest frame is chosen.

        skip_frames (int): number of frames to skip between each
        sampled frame
    """

    def __init__(self,
                 path: str,
                 cache_path: str,
                 num_frames=1,
                 width=640,
                 height=480,
                 fps=30,
                 skip_frames=0):

        super(DoomDataset, self).__init__()
        self.links = load_links(path)
        self.width = width
        self.height = height
        self.num_frames = num_frames
        source_videos = [{
            'id': 'apo9Vb-5pWo',
            'num_frames': 9830,
        }]
        items = []
        total_examples = 0
        for video in source_videos:
            video['num_examples'] = floor(video['num_frames'] / num_frames)
            total_examples += video['num_examples']
        self.source_videos = source_videos
        self.items = items
        self.total_examples = total_examples

    def __getitem__(self, index):
        cur = 0
        for video in self.source_videos:
            end = cur + video['num_examples']
            if index >= end:
                cur = end
                continue
            start_example = index - cur
            start_frame = start_example * self.num_frames
            assert start_frame < video['num_frames']
            return get_frames(video['id'],
                              start_frame,
                              self.num_frames,
                              width=self.width,
                              height=self.height)
        raise ValueError(
            f"unable to seek example, was dataset length calculated incorrectly?")

    def __len__(self):
        return self.total_examples
