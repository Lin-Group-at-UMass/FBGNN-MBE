from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import random
import torch
from run_models import runs
from tool_utils import (
    load_daset_config, 
    load_daset_config_water, 
    load_daset_config_eda
)


def set_seed(seed):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_model(args, device):
    dataset = str(args.dataset).strip()
    n_body = str(args.n_body).strip()
    
    print("="*60)
    print("Configuration Loading")
    print("="*60)
    
    if dataset == 'mix':
        print("Loading water configuration...")
        all_configs = load_daset_config_water()
    elif dataset.startswith('H2O_mix'):
        print("Loading water configuration...")
        all_configs = load_daset_config_water()
    else:
        print("Loading standard configuration...")
        all_configs = load_daset_config()
        
    print(f"Dataset: {dataset}")
    print(f"Available datasets: {list(all_configs.keys())}")
    
    if dataset not in all_configs:
        raise ValueError(f'Invalid dataset name or index: {dataset}')
        
    specific_config = all_configs[dataset]
    specific_config['name'] = dataset
    
    if args.labels:
        labels = args.labels.split(',')
        print(f"Using labels from command line: {labels}")
    else:
        labels = specific_config.get('labels', {}).get(n_body, [])
        print(f"Using labels from config: {labels}")

    if not isinstance(labels, list):
        labels = [labels] if labels else []
    
    if not labels:
        raise ValueError(
            f"No labels specified for dataset {dataset} with n_body={n_body}. "
            f"Please specify labels using --labels argument."
        )

    print(f"\nFinal configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {specific_config['name']}")
    print(f"  N-body: {n_body}")
    print(f"  Labels: {labels}")
    print("="*60 + "\n")
    
    runs(labels, int(n_body), args, device, specific_config['name'])


def run_parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='GPU number.')
    parser.add_argument('--seed', type=int, default=920, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=900, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay value.')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
    parser.add_argument('--n_body', type=str, default="2", help='2 for 2body, 3 for 3body.')
    parser.add_argument('--dataset', type=str, default="3", help='check dataset_config')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    # H2O: 2body 6, 3body 9
    # PhOH: 2body 26, 3body 39
    # PhOH_H2O: 2body [6, 16, 26], 3body [9, 19, 29, 39]
    parser.add_argument('--labels', type=str, default="", help='"" for all labels or "6,16,26" for specific labels')
    parser.add_argument('--cutoff_l', type=float, default=5.0, help='Distance cutoff used in the global layer')
    parser.add_argument('--cutoff_g', type=float, default=5.0, help='Distance cutoff used in the local layer')
    parser.add_argument('--normalized', type=int, default=1, help='Normalized the output energy')
    parser.add_argument('--debug', type=bool, default=False, help='Deubg mode')
    parser.add_argument('--model', type=str, default="VisNet", help='mxmnet, pamnet, pamnet_s, SchNet, DimeNet, '
                                                                    'DimeNet++ and VisNet')
    parser.add_argument('--reduce_val', type=int, default=0, help='Reduce val and test size to 10000')
    parser.add_argument('--er', type=int, default=0, help="early stopping or not")

    args = parser.parse_args()
    return args

def set_see_and_device(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    set_seed(args.seed)
    return device

if __name__ == '__main__':
    args = run_parsers()
    device = set_see_and_device(args)
    train_model(args, device)
