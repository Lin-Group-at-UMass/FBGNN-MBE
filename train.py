from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import random
import torch
from run_models import runs
from tool_utils import load_daset_config


def set_seed(seed):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_model(args, device):
    dataset = args.dataset.lower()
    n_body = args.n_body.lower()
    
    dataset_config = load_daset_config()
    if dataset not in dataset_config.keys():
        raise ValueError('Invalid dataset name')
    
    dataset_config = dataset_config[dataset]
    default_labels = dataset_config['labels']
    str_labels = args.labels
    labels = str_labels.split(',') if str_labels != '' else default_labels[n_body]
    
    runs(labels, int(n_body), args, device, dataset_config['name'])
    
    # if dataset == '3':    
    #     if n_body == '2' or n_body == '3' or n_body == '1':
    #         # adjust this if we like to run only one type of molecular later
    #         default_labels = {
    #             '2': ['6', '16', '26'],
    #             '3': ['9', '19', '29', '39']
    #         }
            
    #         if n_body == '1':
    #             labels_2 = str_labels.split(',') if str_labels != '' else default_labels['2']
    #             labels_3 = str_labels.split(',') if str_labels != '' else default_labels['3']
                
    #             runs(labels_2, 2, args, device, '')
    #             runs(labels_3, 3, args, device, '')
    #         else:
    #             labels = str_labels.split(',') if str_labels != '' else default_labels[n_body]
    #             runs(labels, int(n_body), args, device, '')
    # elif dataset == '1':
    #     labels = ['6'] if n_body == '2' else ['9']
    #     runs(labels, int(n_body), args, device, '_H2O')
    # elif dataset == '4':
    #     labels = ['6'] if n_body == '2' else ['9']
    #     runs(labels, int(n_body), args, device, '_H2O_MP2')
    # elif dataset == '5':
    #     labels = ['6'] if n_body == '2' else ['9']
    #     runs(labels, int(n_body), args, device, '_H2O_2density_MP2')
    # elif dataset == '6':
    #     labels = ['6'] if n_body == '2' else ['9']
    #     runs(labels, int(n_body), args, device, '_H2O_2density_wb97xd3')
    # elif dataset == '2':
    #     labels = ['26'] if n_body == '2' else ['39']
    #     runs(labels, int(n_body), args, device, '_PhOH')
    # else:
    #     raise ValueError('Invalid dataset name')

def run_parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='GPU number.')
    parser.add_argument('--seed', type=int, default=920, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=900, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay value.')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
    parser.add_argument('--n_body', type=str, default="2", help='2 for 2body, 3 for 3body.')
    parser.add_argument('--dataset', type=str, default="3", help='check dataset_config')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    # H2O: 2body 6, 3body 9
    # PhOH: 2body 26, 3body 39
    # PhOH_H2O: 2body [6, 16, 26], 3body [9, 19, 29, 39]
    parser.add_argument('--labels', type=str, default="", help='"" for all labels or "6,16,26" for specific labels')
    parser.add_argument('--cutoff_l', type=float, default=5.0, help='Distance cutoff used in the global layer')
    parser.add_argument('--cutoff_g', type=float, default=15.0, help='Distance cutoff used in the local layer')
    parser.add_argument('--normalized', type=int, default=1, help='Normalized the output energy')
    parser.add_argument('--debug', type=bool, default=False, help='Deubg mode')
    parser.add_argument('--model', type=str, default="mxmnet", help='mxmnet, pamnet, pamnet_s, SchNet, DimeNet, '
                                                                    'DimeNet++')
    parser.add_argument('--reduce_val', type=int, default=0, help='Reduce val and test size to 10000')
    parser.add_argument('--er', type=int, default=0, help="early stopping or not")

    args = parser.parse_args()
    return args

def set_see_and_device(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #if torch.cuda.is_available():
    #    torch.cuda.set_device(args.gpu)
        
    set_seed(args.seed)
    return device

if __name__ == '__main__':
    args = run_parsers()
    device = set_see_and_device(args)
    train_model(args, device)