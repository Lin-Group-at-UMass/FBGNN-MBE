import os
import pathlib
from datetime import datetime
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def make_sure_folder_exists(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%H%M%S")

def make_path(name, with_datetime, with_result):
    if with_datetime:
        name = name_with_datetime(name)
    abs_path = pathlib.Path(__file__).parent.resolve()
    
    sub_path = os.path.dirname(name)
    if with_result:
        file_dir = f'{abs_path}/results/{sub_path}'
    else:
        file_dir = f'{abs_path}/{sub_path}'
    make_sure_folder_exists(file_dir)
    return abs_path, name, file_dir

def save_to_file(content, name, ftype='txt', with_datetime=True, with_result=True):
    abs_path, name, file_dir = make_path(name, with_datetime, with_result)
    
    if ftype == 'json':
        content = json.dumps(content, cls=NpEncoder)
    if with_result:
        file_path = f'{abs_path}/results/{name}.{ftype}'
    else:
        file_path = f'{abs_path}/{name}.{ftype}'   
        
    with open(file_path, 'w') as f:
        f.write(content)
    
    return file_path

def ensure_path_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
