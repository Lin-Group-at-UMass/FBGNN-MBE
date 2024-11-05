import numpy as np
from torch_geometric.data import (InMemoryDataset, Data)
import torch
from structures import mixed_mos


class MixedBodyDS(InMemoryDataset):

    def __init__(self, root, src, labels, max_val, min_val, transform=None, pre_transform=None, pre_filter=None):
        self.src = src
        self.labels = labels
        self.max_val = max_val
        self.min_val = min_val
        root = f"in_memory_data/{root}"
        super(MixedBodyDS, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.src.split('/')[-1]]

    @property
    def processed_file_names(self):
        return self.src.split('/')[-1]

    def download(self):
        pass

    def process(self):
        data_list = []

        suppl = np.load(self.src, allow_pickle=True)
        max_val = self.max_val
        min_val = self.min_val
        
        X = suppl['x']
        Y = suppl['y']
        label = suppl['label']
        
        for i in range(suppl['total']):
            if f'{label[i]}' not in self.labels:
                continue
            mo = mixed_mos[f'{label[i]}']
            x = mo['index']
            edge_index = mo['edges']
            
            y = Y[i]
            if max_val and min_val:
                y = (y - min_val) / (max_val - min_val)
            pos = X[i]
            
            x = torch.tensor(x, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            
            pos = torch.tensor(pos, dtype=torch.float)
            
            data = Data(x=x, pos=pos, edge_index=edge_index, y=y)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

class ApplicationDS(InMemoryDataset):
    def __init__(self, root, src, transform=None, pre_transform=None, pre_filter=None):
        self.src = src
        root = f"in_memory_data/{root}"
        super(ApplicationDS, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.root}.pt']

    def download(self):
        pass

    def process(self):
        data_list = []

        suppl = np.load(self.src, allow_pickle=True)

        X = suppl['x']
        Y = suppl['y']
        label = suppl['label']

        print("Available keys in mixed_mos:", mixed_mos.keys())  # Debugging

        for i in range(suppl['total']):
            label_str = ' '.join(map(str, label[i]))  # Ensure consistent format
            label_test = [float(i) for i in label[i].split()]
            print(f"Processing label: {label_str}")  # Debugging

            label_key = 0
            for key, value in zip(mixed_mos.keys(), mixed_mos.values()):
                if label_test == value['index']:
                    label_key = key
                    break
                else:
                    print(f"Label {label_str} not found in mixed_mos keys.")  # Debugging
                    continue

            mo = mixed_mos[label_key]
            x = mo['index']
            edge_index = mo['edges']

            y = Y[i]
            pos = X[i]

            x = torch.tensor(x, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            pos = torch.tensor(pos, dtype=torch.float)
            y = torch.tensor([y], dtype=torch.float)  # Ensure y is a tensor

            data = Data(x=x, pos=pos, edge_index=edge_index, y=y)
            data_list.append(data)

        if not data_list:
            raise ValueError("No data matched the labels in mixed_mos. Please check the label format and contents.")

        torch.save(self.collate(data_list), self.processed_paths[0])
