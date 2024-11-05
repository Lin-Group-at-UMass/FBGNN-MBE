from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SchNet, DimeNetPlusPlus, DimeNet, ViSNet, GraphUNet
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from utils import EMA
from model import MXMNet, Config, Config_GNN
from file_utils import save_to_file, name_with_datetime
from mixed_body_dataset import MixedBodyDS
from tqdm import tqdm
from pamnet import PAMNet, PAMNet_s
import time
from tool_utils import log_de_normalized
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def test(loader, ema, model, device):
    total_error = 0
    total_squared_error = 0
    sum_of_squares = 0
    total_data = 0

    ema.assign(model)

    y_predict = []
    y_true = []
    for data in loader:
        data = data.to(device)
        if test_models:
            z = data.x.squeeze()
            pos = data.pos
            batch = getattr(data, 'batch', None)
            output = model(z, pos, batch).view(-1)
        else:
            output = model(data)
        y_predict.append(output.detach().cpu().numpy())
        y_true.append(data.y.cpu().numpy())
        
    ema.resume(model)
    
    y_predict = np.concatenate(y_predict, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    
    y_predict = log_de_normalized(y_predict, loader.dataset.max_val, loader.dataset.min_val)
    y_true = log_de_normalized(y_true, loader.dataset.max_val, loader.dataset.min_val)
    
    mse = mean_squared_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    r2 = r2_score(y_true, y_predict)
    
    return {
        'loss': mae,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'y_predict':  y_predict.tolist(),
        'y_true': y_true.tolist()
    }

def msd(y_true, y_pred):
    return (y_pred - y_true).mean()

def test_formse(y_predict, y_true, max_val, min_val):

    # y_predict = np.concatenate(y_predict, axis=0)
    # y_true = np.concatenate(y_true, axis=0)

    y_predict = log_de_normalized(y_predict, max_val, min_val)
    y_true = log_de_normalized(y_true, max_val, min_val)

    mse = msd(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    r2 = r2_score(y_true, y_predict)

    return {
        'loss': mae,
        'mse': mse,
        'mae': mae,
        'r2': r2,
    }

def get_model(args, device):
    config = Config(dim=args.dim, 
                    n_layer=args.n_layer,
                    cutoff_l=args.cutoff_l,
                    cutoff_g=args.cutoff_g)
    
    model = args.model
    print("XXXXXX",model)
    global test_models
    test_models = False
    if model == 'pamnet':
        test_models = False
        model = PAMNet(config).to(device)
    elif model == 'pamnet_s':
        test_models = False
        model = PAMNet_s(config).to(device)
    elif model == 'SchNet':
        test_models = True
        model = SchNet(hidden_channels=args.dim,
                        num_interactions=args.n_layer,
                        cutoff=args.cutoff_g,).to(device)
    elif model == 'DimeNet':
        test_models = True
        model = DimeNet(hidden_channels=args.dim,
                        out_channels=1,
                        num_blocks=args.n_layer,
                        cutoff=args.cutoff_g,
                        num_bilinear=8,
                        num_spherical=7,
                        num_radial=6,).to(device)
    elif model == 'DimeNet++':
        test_models = True
        model = DimeNetPlusPlus(hidden_channels=args.dim,
                                out_channels=1,
                                num_blocks=args.n_layer,
                                cutoff=args.cutoff_g,
                                int_emb_size=64,
                                basis_emb_size=8,
                                out_emb_channels=256,
                                num_spherical=7,
                                num_radial=6,).to(device)
    elif model == 'ViSNet':
        test_models = True
        print("xxxxxx")
        model = ViSNet(hidden_channels=args.dim,
                       num_layers=args.n_layer,
                       cutoff=args.cutoff_g).to(device)
    else:
        test_models = False
        model = MXMNet(config).to(device)
    print(test_models)
    print(f'Loaded model: {args.model}')
    return model

def get_folder(args):
    folder = f'results/ds_{args.dataset}/{args.model}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def run_model(train_loader, test_loader, val_loader, train_dataset_len, args, device, name):
    model = get_model(args, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

    ema = EMA(model, decay=0.999)

    print('===================================================================================')
    print('                                Start training')
    print('===================================================================================')

    best_epoch = None
    best_val_loss = None
    mse = None
    r2 = None
    y_predict = None
    y_true = None
    
    logs = []

    early_stopper = EarlyStopper(patience=10, min_delta=0.000001)
    if args.debug:
        args.epochs = 5
    print(22222)
    for epoch in tqdm(range(args.epochs)):
        loss_all = 0
        step = 0
        model.train()

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            print(test_models)
            if test_models:
                print(33333)
                z = data.x.squeeze()
                pos = data.pos
                batch = getattr(data, 'batch', None)
                output = model(z, pos, batch).view(-1)
                print(44444)
            else:
                print(3333)
                output = model(data)
                print(4444)
            print(55555)
            loss = F.l1_loss(output, data.y)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
            optimizer.step()

            curr_epoch = epoch + float(step) / (train_dataset_len / args.batch_size)
            scheduler_warmup.step(curr_epoch)

            ema(model)
            step += 1

        train_loss = loss_all / len(train_loader.dataset)

        val_result = test(val_loader, ema, model, device)
        val_loss = val_result['loss']

        if best_val_loss is None or val_loss <= best_val_loss:
            test_result = test(test_loader, ema, model, device)
            test_loss = test_result['loss']
            mse = test_result['mse']
            mae = test_result['mae']
            r2 = test_result['r2']
            best_epoch = epoch
            best_val_loss = val_loss
            y_predict = test_result['y_predict']
            y_true = test_result['y_true']

        print('Epoch: {:03d}, Train MAE: {:.7f}, Validation MAE: {:.7f}, '
            'Test MAE: {:.7f}'.format(epoch+1, train_loss, val_loss, test_loss))
        print('\tTest MSE: {:.7f}, Test R2: {:.7f}'.format(mse, r2))
        
        log = {
            'time': time.time(),
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'mae': mae,
            'mse': mse,
            'r2': r2
        }
        
        if early_stopper.early_stop(val_loss):             
            break
    
    print('===================================================================================')
    print('Best Epoch:', best_epoch)
    print('Best Test MAE:', test_loss)
    print('MSE:', mse)
    print('R2:', r2)
    
    results = {
        'best_epoch': best_epoch ,
        'test_loss': test_loss,
        'mse': mse,
        'mae': test_loss,
        'r2': r2,
        'y_predict': y_predict,
        'y_true': y_true,
        'timestap': time.time()
    }

    save_to_file(logs, f'{get_folder(args)}/train_logs_{name}', 'json', with_datetime=False, with_result=False)
    torch.save(model.state_dict(), f'{get_folder(args)}/model_{name}.pt')
    
    return results

def get_min_max(labels, train_path, test_path, val_path):
    # may need to be adjusted with labels
    # need to be refactored
    train = np.load(train_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    val = np.load(val_path, allow_pickle=True)
    
    max_val, min_val = -100000, 100000

    for label in labels:
        label = int(label)
        train_y = train['y']
        test_y = test['y']
        val_y = val['y']
        train_y = train_y[train['label'] == label]
        test_y = test_y[test['label'] == label]
        val_y = val_y[val['label'] == label]
        
        train_max_val, trian_min_val = np.max(train_y), np.min(train_y)
        test_max_val, test_min_val = np.max(test_y), np.min(test_y)
        val_max_val, val_min_val = np.max(val_y), np.min(val_y)
        
        t_max_val = max(train_max_val, test_max_val, val_max_val)
        t_min_val = min(trian_min_val, test_min_val, val_min_val)
        
        max_val = max(max_val, t_max_val)
        min_val = min(min_val, t_min_val)     
    return max_val, min_val

def runs(labels, n_body, args, device, dataset_name):
    train_path = f"../dataset/{dataset_name}/{n_body}body_energy_train.npz"
    test_path = f"../dataset/{dataset_name}/{n_body}body_energy_test.npz"
    val_path = f"../dataset/{dataset_name}/{n_body}body_energy_val.npz"
    # check min max first
    key = "_".join(labels)
    key = f"{key}_{n_body}_body"
    
    max_val, min_val = None, None
    if args.normalized:
        max_val, min_val = get_min_max(labels, train_path, test_path, val_path)
    else:
        key = f"{key}_non_normalized"
    
    train_dataset = MixedBodyDS(f'train_{key}', train_path, labels, max_val, min_val)
    test_dataset = MixedBodyDS(f'test_{key}', test_path, labels, max_val, min_val)
    val_dataset = MixedBodyDS(f'val_{key}', val_path, labels, max_val, min_val)
    
    key = f"{key}_{args.model}"
    print(key, '==>', train_dataset.max_val, train_dataset.min_val, args.debug)
    
    if args.debug:
        train_dataset = train_dataset[0: 100]
        val_dataset = val_dataset[0: 100]
        test_dataset = test_dataset[0: 100]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"{key} ==> train:{len(train_dataset)}, val:{len(val_dataset)}, test:{len(test_dataset)}")
    start_time = time.time()
    
    key = name_with_datetime(key)
    print(11111)
    results = run_model(train_loader, test_loader, val_loader, len(train_dataset), args, device, key)
    print(22222)
    end_time = time.time()
    
    results['duration'] = end_time - start_time
    results['start_time'] = start_time
    results['end_time'] = end_time
    results['min_val'] = min_val
    results['max_val'] = max_val
    results['args'] = args.__dict__
    np.save(f'{get_folder(args)}/train_best_{key}', results)
    
    return results
