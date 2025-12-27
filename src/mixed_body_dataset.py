import os
import numpy as np
from torch_geometric.data import (InMemoryDataset, Data)
import torch
import torch.nn.functional as F
from structures import mixed_mos


class MixedBodyDS(InMemoryDataset):
    def __init__(self, root, src, labels, max_val, min_val, transform=None, pre_transform=None, pre_filter=None):
        self.src = src
        self.labels = labels
        self.max_val = max_val
        self.min_val = min_val
        processed_root = f"in_memory_data/{root}"
        super(MixedBodyDS, self).__init__(processed_root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"Loaded processed data from: {self.processed_paths[0]}")

    @property
    def raw_file_names(self):
        return [os.path.basename(self.src)]

    @property
    def raw_dir(self):
        return os.path.dirname(self.src)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data_list = []
        source_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        suppl = np.load(source_path, allow_pickle=True)
        
        max_v = self.max_val
        min_v = self.min_val
        
        X = suppl['x']
        if 'y_energy' in suppl.files:
            Y = suppl['y_energy']
        elif 'y' in suppl.files:
            Y = suppl['y']
        else:
            raise KeyError(f"Neither 'y' nor 'y_energy' found in {self.raw_paths[0]}.")
        all_labels = suppl['label']
        
        print(f"Processing {suppl['total']} data points from {self.src}...")
        
        label_mask = np.array([str(l) in self.labels for l in all_labels])
        Y_filtered = Y[label_mask]
        
        if len(Y_filtered) == 0:
            raise ValueError(f"No data found for labels {self.labels}")
        
        y_mean = np.mean(Y_filtered)
        y_std = np.std(Y_filtered)
        
        if max_v is not None and min_v is not None:
            y_range = max_v - min_v
            if y_range < 1e-8:
                norm_mode = 'zscore'
            else:
                norm_mode = 'minmax'
        elif y_std > 1e-8:
            norm_mode = 'zscore'
        else:
            norm_mode = 'none'
        
        for i in range(suppl['total']):
            current_label = str(all_labels[i])
            if current_label not in self.labels:
                continue

            mo = mixed_mos[current_label]
            x = mo['index']
            edge_index = mo['edges']
            y = Y[i]
            pos = X[i]

            if norm_mode == 'minmax':
                y_normalized = (y - min_v) / (max_v - min_v)
            elif norm_mode == 'zscore':
                y_normalized = (y - y_mean) / y_std
            else:
                y_normalized = y

            data = Data(
                x=torch.tensor(x, dtype=torch.long),
                pos=torch.tensor(np.asarray(pos, dtype=np.float32), dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor(float(y_normalized), dtype=torch.float32)
            )
            data_list.append(data)

        if not data_list:
            raise ValueError("No data was processed.")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed {len(data_list)} samples")


class MixedBodyGradient:
    def __init__(self, root, src, labels, energy_mean=0.0, energy_std=1.0,
                 forces_mean=0.0, forces_std=1.0,
                 transform=None, pre_transform=None, pre_filter=None):
        self.src = src
        self.labels = [str(l) for l in labels]
        self.energy_mean = energy_mean
        self.energy_std = energy_std
        self.forces_mean = forces_mean
        self.forces_std = forces_std
        self.transform = transform
        
        cache_dir = "in_memory_data/gradient_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        src_basename = os.path.basename(src).replace('.npz', '')
        labels_str = '_'.join(sorted(self.labels))
        self._cache_file = os.path.join(cache_dir, f"{src_basename}_{labels_str}.pt")
        
        if os.path.exists(self._cache_file):
            print(f"[MixedBodyGradient] Loading from cache: {self._cache_file}")
            cache_data = torch.load(self._cache_file)
            self.data_list = cache_data['data_list']
            self.energy_mean = cache_data.get('energy_mean', energy_mean)
            self.energy_std = cache_data.get('energy_std', energy_std)
            self.forces_mean = cache_data.get('forces_mean', forces_mean)
            self.forces_std = cache_data.get('forces_std', forces_std)
            print(f"[MixedBodyGradient] Loaded {len(self.data_list)} samples from cache")
        else:
            print(f"[MixedBodyGradient] Cache not found, processing {src}...")
            self.data_list = self._process()
            torch.save({
                'data_list': self.data_list,
                'energy_mean': self.energy_mean,
                'energy_std': self.energy_std,
                'forces_mean': self.forces_mean,
                'forces_std': self.forces_std,
            }, self._cache_file)
            print(f"[MixedBodyGradient] Saved {len(self.data_list)} samples to {self._cache_file}")

    def _process(self):
        suppl = np.load(self.src, allow_pickle=True)
        
        coords_all = suppl['x']
        energy_all = suppl['y']
        forces_all = suppl['f']
        labels_all = suppl['labels']
        atom_types_all = suppl['atom_types']
        
        label_set = set(self.labels)
        valid_indices = [i for i, lbl in enumerate(labels_all) if str(lbl) in label_set]
        
        if not valid_indices:
            raise ValueError(f"No matching labels found.")
        
        print(f"[MixedBodyGradient] Found {len(valid_indices)} matching samples")
        
        E_mean, E_std = self.energy_mean, self.energy_std
        F_mean, F_std = self.forces_mean, self.forces_std
        
        edge_cache = {}
        data_list = []
        
        for i in valid_indices:
            label_str = str(labels_all[i])
            
            if label_str not in edge_cache:
                mo = mixed_mos.get(label_str)
                if mo is None:
                    continue
                edge_cache[label_str] = torch.tensor(mo['edges'], dtype=torch.long)
            
            energy_norm = (energy_all[i] - E_mean) / E_std
            forces_norm = (forces_all[i] - F_mean) / F_std
            
            data = Data(
                x=torch.from_numpy(atom_types_all[i].astype(np.int64)),
                edge_index=edge_cache[label_str],
                pos=torch.from_numpy(coords_all[i].astype(np.float32)),
                y=torch.tensor([[energy_norm]], dtype=torch.float32),
                f=torch.from_numpy(forces_norm.astype(np.float32))
            )
            data_list.append(data)
        
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


class MixedBodyEDA(InMemoryDataset):
    def __init__(self, root, src, labels, mean_val=None, std_val=None,
                 use_charge=False,
                 transform=None, pre_transform=None, pre_filter=None):
        self.src = src
        self.labels = set(map(str, labels))
        self.mean_val = mean_val
        self.std_val = std_val
        self.use_charge = use_charge

        suffix = "_charge" if self.use_charge else ""
        root = f"in_memory_data/{root}{suffix}"
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.src.split('/')[-1]] if self.src else []

    @property
    def processed_file_names(self):
        if not self.raw_file_names:
            return []
        raw_name = self.raw_file_names[0]
        suffix = "_charge" if self.use_charge else ""
        return [raw_name.replace('.npz', f'{suffix}.pt')]

    def download(self):
        pass

    def process(self):
        suppl = np.load(self.src, allow_pickle=True)
        
        X, labels_arr = suppl['x'], suppl['labels']
        Y = np.stack([suppl['frz'], suppl['pol'], suppl['ct']], axis=1)
        
        has_charges = 'charges' in suppl.keys()
        Charges = suppl['charges'] if has_charges else None

        data_list = []
        print(f"[INFO] Processing {len(X)} total samples from {self.src}")

        for i in range(len(X)):
            lab = str(labels_arr[i])
            if self.labels and lab not in self.labels:
                continue
            
            mo = mixed_mos.get(lab)
            if mo is None:
                continue

            pos = torch.from_numpy(X[i].astype(np.float32))
            x = torch.tensor(mo['index'], dtype=torch.long)
            edge_index = torch.tensor(mo['edges'], dtype=torch.long)
            y = Y[i]

            if not np.all(np.isfinite(y)):
                continue

            if self.mean_val is not None and self.std_val is not None:
                mean_v = self.mean_val[:3]
                std_v = self.std_val[:3]
                std_v[std_v == 0] = 1.0
                y = (y - mean_v) / std_v

            data = Data(x=x, pos=pos, edge_index=edge_index, y=torch.tensor(y, dtype=torch.float))

            if self.use_charge:
                if not has_charges or Charges is None:
                    raise ValueError(f"'charges' not found in {self.src}")
                data.charges = torch.from_numpy(np.asarray(Charges[i], dtype=np.float32)).view(-1, 1)

            data_list.append(data)

        if not data_list:
            raise ValueError("No data was processed.")

        torch.save(self.collate(data_list), self.processed_paths[0])
        print(f"Processed data saved to: {self.processed_paths[0]}")


class ApplicationDS(InMemoryDataset):
    def __init__(self, root, src, labels, mean_val=None, std_val=None, 
                 transform=None, pre_transform=None, pre_filter=None):
        self.src = src
        self.labels = labels
        self.mean_val = mean_val
        self.std_val = std_val
        root = f"in_memory_data/{root}"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.src.split('/')[-1]]

    @property
    def processed_file_names(self):
        return [self.raw_file_names[0].replace('.npz', '.pt')]

    def download(self):
        pass

    def process(self):
        suppl = np.load(self.src, allow_pickle=True)
        
        X = suppl['x']
        Y = suppl.get('y_energy', suppl.get('y'))
        all_labels = suppl.get('label', suppl.get('labels'))
        
        data_list = []
        labels_to_keep = set(self.labels) if self.labels else set()
        mixed_mos_str = {str(k): v for k, v in mixed_mos.items()}

        for i in range(len(X)):
            try:
                label_int = int(all_labels[i])
            except ValueError:
                continue
            
            if labels_to_keep and label_int not in labels_to_keep:
                continue

            label_str = str(label_int)
            if label_str not in mixed_mos_str:
                continue
            
            mo = mixed_mos_str[label_str]
            x = torch.tensor(mo['index'], dtype=torch.long)
            edge_index = torch.tensor(mo['edges'], dtype=torch.long)
            pos = torch.from_numpy(X[i].astype(np.float32))
            
            y = Y[i]
            if self.mean_val is not None and self.std_val is not None and self.std_val > 1e-8:
                y = (y - self.mean_val) / self.std_val

            data = Data(x=x, pos=pos, edge_index=edge_index, y=torch.tensor(float(y), dtype=torch.float32))
            data_list.append(data)

        if not data_list:
            raise ValueError("No valid data processed.")

        torch.save(self.collate(data_list), self.processed_paths[0])
        print(f"Processed {len(data_list)} samples")
