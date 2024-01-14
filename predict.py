import ast
import os

from subaligner.predictor import Predictor

data_file_path = ast.literal_eval(os.environ.get('dataPath'))['path']

predictor = Predictor()
predictor.predict_single_pass(data_file_path=data_file_path)
