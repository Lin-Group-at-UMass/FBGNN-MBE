import numpy as np
from sklearn.model_selection import train_test_split
import os
from file_utils import ensure_path_exists
from tool_utils import load_daset_config

count_dict = {}
types = {'H': 0, 'C': 1, 'O': 2}

def extract_molecule_data_to_numpy(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    molecule_data = []
    count = 0
    indexes = []
    for line in lines:
        if 'C' in line or 'H' in line or 'O' in line:
            components = line.split()
            molecule_data.append(components[1:])
            count += 1
            indexes.append(types[components[0]])
    molecule_data = np.array(molecule_data, dtype=float)
    
    return molecule_data, count


def save_to_file(x, y, labels, name):
    ensure_path_exists(name)
    
    structured_data = {
        'x': x,
        'y': y,
        'label': labels,
        'total': len(x)
    }
    
    np.savez(name, **structured_data)


def get_in_path(folder, n, key):
    f_path = f'{folder}/n_{n}Body/{key}/{key}.in'
    if os.path.isfile(f_path):
        return f_path
    
    f_path = f'{folder}/n_{n}Body/{key[2:]}/{key[2:]}.in'
    if os.path.isfile(f_path):
        return f_path
    
    f_path = f'{folder}/n_{n}Body/{key}.in'
    if os.path.isfile(f_path):
        return f_path
    
    f_path = f'{folder}/n_{n}Body/{key[2:]}.in'
    if os.path.isfile(f_path):
        return f_path
        
    return None

def load_data_from_file(n, folder):
    name = f'{folder}/{n}body_energy.txt'
    
    x = []
    y = []
    labels = []
    with open(name) as f:
        l = f.readlines()
        for line in l:
            line = line.strip('\n')
            energy = line.split(',')[1]
            key = line.split(',')[0]

            f_path = get_in_path(folder, n, key)
             
            if f_path is not None:
                data, label = extract_molecule_data_to_numpy(f_path)                
                labels.append(label)
                x.append(data)
                y.append(float(energy))
            else:
                print(f'File {f_path} not found')
                
    print(f'Loaded {n} body data')
    print(f"Total {len(x)}")
    
    print(folder)
    try:
        x = np.array(x)
    except:
        x = np.array(x, dtype=object)
    y = np.array(y)
    labels = np.array(labels)
    return x, y, labels


def make_body(n, s_folder, d_folder):
    x, y, labels = load_data_from_file(n, s_folder)
    
    X_train, X_temp, y_train, y_temp, label_train, label_temp = train_test_split(x, y, 
                                                                                 labels, 
                                                                                 test_size=0.2, 
                                                                                 random_state=42, 
                                                                                 stratify=labels)
    
    X_test, X_val, y_test, y_val, label_test, label_val = train_test_split(X_temp, y_temp,
                                                                           label_temp, 
                                                                            test_size=0.25, 
                                                                            random_state=42,
                                                                            stratify=label_temp)
    
    save_to_file(X_train, y_train, label_train, f'{d_folder}/{n}body_energy_train.npz')
    save_to_file(X_test, y_test, label_test, f'{d_folder}/{n}body_energy_test.npz')
    save_to_file(X_val, y_val, label_val, f'{d_folder}/{n}body_energy_val.npz')

def make_datasets(s_folder, d_folder):
    make_body(2, s_folder, d_folder)
    make_body(3, s_folder, d_folder)

if __name__ == '__main__':
    
    dataset_config = load_daset_config()
    for key in dataset_config.keys():
        d_cnofig = dataset_config[key]
        make_datasets(d_cnofig['source_path'], f"../dataset/{d_cnofig['name']}")
    
    '''
        for example
        data = np.load(f'{folder}/2body_energy_test_dataset.npz', allow_pickle=True)
        print(data['x'].shape)
        print(data['y'].shape)
        print(data['label'].shape)
        print(data['total'])
    '''