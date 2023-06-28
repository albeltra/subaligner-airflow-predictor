import ast
import os
from pathlib import Path

from subaligner.predictor import Predictor

media_path = ast.literal_eval(os.environ.get('mediaFile'))['path']

subtitle_path = "/audio-subs/" + Path(media_path).name + '.srt'
audio_file_path = "/audio-subs/" + Path(media_path).name + '.wav'
video_file_path = "/data/v4/" + Path(media_path).name

predictor = Predictor()
predictor.predict_single_pass(video_file_path=video_file_path,
                              subtitle_file_path=subtitle_path,
                              audio_file_path=audio_file_path) 
