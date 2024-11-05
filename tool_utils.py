
import json
import numpy as np
def log_de_normalized(arr, max_val, min_val):
    arr = np.array(arr)
    arr = arr * (max_val - min_val) + min_val
    return arr

def load_daset_config():
    data = json.load(open('dataset_config.json'))
    return data