from __future__ import division
from __future__ import print_function

import glob
import os
import numpy as np
from train import set_see_and_device
import argparse
from run_models import get_model, test
from utils import EMA
from tool_utils import load_daset_config
from mixed_body_dataset import MixedBodyDS
import torch
from torch_geometric.loader import DataLoader

def restore_test_results(args, device, weight_path, min_val, max_val):
    dataset = args.dataset.lower()
    n_body = args.n_body.lower()
    
    dataset_config = load_daset_config()
    if dataset not in dataset_config.keys():
        raise ValueError('Invalid dataset name')
    
    dataset_config = dataset_config[dataset]
    default_labels = dataset_config['labels']
    str_labels = args.labels
    labels = str_labels.split(',') if str_labels != '' else default_labels[n_body]

    labels, n_body, args, device, dataset_name = labels, int(n_body), args, device, dataset_config['name']

    train_path = f"../dataset/{dataset_name}/{n_body}body_energy_train.npz"
    test_path = f"../dataset/{dataset_name}/{n_body}body_energy_test.npz"
    val_path = f"../dataset/{dataset_name}/{n_body}body_energy_val.npz"

    key = "_".join(labels)
    key = f"{key}_{n_body}_body"
    
    test_dataset = MixedBodyDS(f'test_{key}', test_path, labels, max_val, min_val)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = get_model(args, device)
    model.load_state_dict(torch.load(weight_path))
    ema = EMA(model, decay=0.999)
    model.eval()
    
    results = test(test_loader, ema, model, device)
    print("results r2=====>", results['r2'])
    
    best_path = weight_path.replace('pt', 'npy').replace('model', 'train_best')
    best_data = np.load(best_path, allow_pickle=True).item()
    best_data['y_predict'] = results['y_predict']
    best_data['y_true'] = results['y_true']
    
    best_path = best_path.replace('train_best', 'restore_train_best')
    np.save(f'{best_path}', best_data)

def get_all_pts():
    folder = f'./results'
    results = []
    paths = glob.glob(os.path.join(folder, "**", '*.pt'), recursive=True)
    paths.sort()
    return paths

if __name__ == '__main__':
    model_states = get_all_pts()
    for ms in model_states:        
        best_path = ms.replace('pt', 'npy').replace('model', 'train_best')
        best_data = np.load(best_path, allow_pickle=True).item()
        
        min_val, max_val = best_data['min_val'], best_data['max_val']
        
        # if (best_data['y_true'] is None):
        #     print(ms)
        print("original r2=====>", best_data['r2'])
        
        args = argparse.Namespace(**best_data['args'])
        device = set_see_and_device(args)       
        restore_test_results(args, device, ms, min_val, max_val)
    
