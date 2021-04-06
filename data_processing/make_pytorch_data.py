import torch
from torch.utils.data import DataLoader
from jetdatasets import JetDataset

def initialize_data(path, batch_size, num_train, num_test=-1, num_val=-1):

    data = torch.load(path)

    # Calculate node masks and edge masks
    if 'labels' in data:
        node_mask = data['labels']
        node_mask = node_mask.to(torch.uint8)
    else:
        node_mask = data['p4'][...,0] != 0
        node_mask = node_mask.to(torch.uint8)
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    data['node_mask'] = node_mask
    data['edge_mask'] = edge_mask

    jet_data = JetDataset(data, shuffle=True) # The original data is not shuffled yet

    if not (num_test < 0 or num_val < 0): # Specified num_test and num_val
        assert num_train + num_test + num_val <= len(jet_data), f"num_train + num_test + num_val = {num_train + num_test + num_val} \
                                                                is larger than the data size {len(jet_data)}!"

        # split into training, testing, and valid set
        jet_data = JetDataset(jet_data[0: num_train + num_test + num_val], shuffle = True)
        train_set, test_set, valid_set = torch.utils.data.random_split(jet_data, [num_train, num_test, num_val])
        train_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=False)

    # Unspecified num_test and num_val -> Choose training data and then divide the rest in half into testing and validation datasets
    else:
        assert num_train <= len(jet_data), f"num_train = {num_train} is larger than the data size {len(jet_data)}!"

        # split into training, testing, and valid sets
        # split the rest in half
        num_test = int((len(jet_data) - num_train) / 2)
        num_val = len(jet_data) - num_train - num_test
        train_set, test_set, valid_set = torch.utils.data.random_split(jet_data, [num_train, num_test, num_val])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, valid_loader
