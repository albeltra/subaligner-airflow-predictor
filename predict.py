import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-m', help='Description for bar argument', required=True)
parser.add_argument('-v', help='Description for bar argument', required=True)
parser.add_argument('-c', help='Description for bar argument', required=True)
parser.add_argument('-s', help='Description for bar argument', required=True)
args = vars(parser.parse_args())

alignment_level = args.get('m')
media_path = args.get('v')
subtitle_track = args.get('s')
audio_channel = args.get('c')
print(media_path)
print(subtitle_track)

command = ["subaligner",
           "-m",
           alignment_level,
           "-v",
           media_path,
           "-s",
           subtitle_track,
           "-c",
           audio_channel,
           "-o",
           "/tmp/" + Path(media_path).name + ".srt"]

subprocess.run(command, check=True)