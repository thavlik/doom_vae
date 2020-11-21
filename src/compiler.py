import argparse
import json
import youtube_dl
import os

parser = argparse.ArgumentParser(
    description='Youtube dataset compiler')
parser.add_argument('--input',  '-i',
                    dest="input",
                    metavar='INPUT',
                    help='path to text file containing youtube video or playlist links',
                    default='../dataset/links.txt')
parser.add_argument('--output',
                    dest="output",
                    metavar='OUTPUT',
                    help='output file path',
                    default='../dataset/compiled.json')
parser.add_argument('--download',
                    dest="download",
                    metavar='DOWNLOAD',
                    help='download videos if true',
                    default=False)
parser.add_argument('--cache_dir',
                    dest="cache_dir",
                    metavar='CACHE_DIR',
                    help='video download path',
                    default='../dataset/cache')
args = parser.parse_args()


completed = {}
videos = []

completed_path = os.path.join(os.path.dirname(args.output), '.completed.txt')

try:
    with open(completed_path) as f:
        completed = json.loads(f.read())
except:
    pass

if os.path.exists(args.output):
    with open(args.output) as f:
        videos = json.loads(f.read())


def write_completed():
    with open(completed_path, 'w') as f:
        f.write(json.dumps(completed))


def write_videos():
    with open(args.output, 'w') as f:
        f.write(json.dumps(videos))


def process_video(video):
    videos.append({
        k: video[k] for k in ['id',
                              'ext',
                              'vcodec',
                              'uploader_id',
                              'channel_id',
                              'duration',
                              'width',
                              'height',
                              'fps']
    })


with open(args.input, "r") as f:
    lines = [line.strip() for line in f]

with youtube_dl.YoutubeDL({
    'outtmpl': '%(id)s.%(ext)s',
    'cachedir': args.cache_dir,
}) as ydl:
    for line in lines:
        if line in completed:
            continue
        result = ydl.extract_info(
            line,
            download=args.download,
        )
        if 'entries' in result:
            # It is a playlist
            for video in result['entries']:
                process_video(video)
        else:
            # Just a single video
            process_video(result)
        write_videos()
        completed[line] = True
        write_completed()
