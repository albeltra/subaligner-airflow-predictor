import ast
import subprocess
import os
from pathlib import Path
from subaligner.predictor import Predictor

media_path = ast.literal_eval(os.environ.get('mediaFile'))['path']

subtitle_path = "/TEMP-SUBS/" + Path(media_path).name + '.srt'
audio_file_path = "/TEMP-SUBS/" + Path(media_path).name + '.wav'

predictor = Predictor()
predictor.predict_single_pass(video_file_path='',
                              subtitle_file_path=subtitle_path,
                              audio_file_path=audio_file_path)
