import os
import glob
import numpy as np
import json

def get_all_reresults():
    folder = f'./best_results'
    
    results = []
    paths = glob.glob(os.path.join(folder, "**", '*.npy'), recursive=True)
    paths.sort()
    return paths


def display_results():
    paths = get_all_reresults()

    ds = json.load(open('dataset_config.json'))
    for p in paths:
        data = np.load(p, allow_pickle=True)
        
        keys = ['r2', 'best_epoch', 'test_loss', 'duration']
        items = data.item()
        
        strings = ''
        for key in keys:
            strings += f'{key}:{items[key]:.4f}, '
        
        net = 'mxmnet' if 'mxmnet' in p else 'pamnet'
        args = items['args']
        cutoff_l = args['cutoff_l']
        cutoff_g = args['cutoff_g']
        epochs = args['epochs']
        n_body = args['n_body']
        dataset = args['dataset']
        dataset = ds[dataset]['name']
        print(len(items['y_predict']))
        print(p)
        print(f'{net}, {dataset} n_body: {n_body} cutoff_l: {cutoff_l}, cutoff_g: {cutoff_g}, total epochs: {epochs}')
        print(strings)
        print('')

if __name__ == '__main__':
    display_results()
