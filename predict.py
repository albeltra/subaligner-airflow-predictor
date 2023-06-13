import ast
import subprocess
import os
from pathlib import Path

media_path = ast.literal_eval(os.environ.get('mediaFile'))['path'] 

subtitle_path = "/TEMP-SUBS/" + Path(media_path).name + '.srt'
audio_file_path = "/TEMP-SUBS/" + Path(media_path).name + '.wav'

command = ["subaligner",
           "-m",
           'single',
           "-v",
           media_path,
           "-s",
           subtitle_path,
           "-a",
           audio_file_path,]

subprocess.run(command, check=True)
