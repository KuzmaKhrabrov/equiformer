from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np

from ase.db import connect

import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
from torch.utils.data import Subset


class NablaDFT(InMemoryDataset):
    def __init__(self, root, split="train2k", transform=None, pre_transform=None):
        
        # For simplicity, always use one type of molecules 
        '''
        if dataset_arg == "all":
            dataset_arg = ",".join(MD17.available_molecules)
        '''
        

        '''
        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )
        '''
        #saelf.raw_file_names = [f'/mnt/2tb/khrabrov/schnet/data/{split}_v2_formation_energy_w_forces.db']
        self.split = split
        super(NablaDFT, self).__init__(root, transform, pre_transform)
        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )

    @property
    def processed_file_names(self):
        return f'{self.split}.pt'
            
    @property
    def raw_file_names(self):
        return [f'/mnt/2tb/khrabrov/schnet/data/{self.split}_v2_formation_energy_w_forces.db']
    
    def len(self):
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(NablaDFT, self).get(idx - self.offsets[data_idx])

    #def download(self):
    #    return
    #    raise NotImplementedError

    def process(self):
        db = connect(self.raw_file_names[0])
        samples = []
        for db_row in tqdm(db.select()):
            z = torch.from_numpy(db_row.numbers).long()
            positions = torch.from_numpy(db_row.positions).float()
            y = torch.from_numpy(np.array(db_row.data['energy'])).float()
            if "forces" in db_row.data:
                dy = torch.from_numpy(np.array(db_row.data['forces'])).float()
            else:
                dy = torch.zeros(1)

            samples.append(Data(z=z, pos=positions, y=y, dy=dy))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])


# From https://github.com/torchmd/torchmd-net/blob/72cdc6f077b2b880540126085c3ed59ba1b6d7e0/torchmdnet/utils.py#L54
def train_val_test_split(dset_len, train_size, val_size, test_size, seed, order=None):
    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        print(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=np.int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


# From: https://github.com/torchmd/torchmd-net/blob/72cdc6f077b2b880540126085c3ed59ba1b6d7e0/torchmdnet/utils.py#L112
def make_splits(
    dataset_len,
    train_size,
    val_size,
    test_size,
    seed,
    filename=None,  # path to save split index
    splits=None,
    order=None,
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_size, val_size, test_size, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


def get_nablaDFT_datasets(root, 
    split, train_size, val_size, test_size, 
    seed):
    '''
        Return training, validation and testing sets of MD17 with the same data partition as TorchMD-NET.
    '''

    all_dataset = NablaDFT(root, split=split)

    idx_train, idx_val, idx_test = make_splits(
        len(all_dataset),
        train_size, val_size, test_size, 
        seed, 
        filename=os.path.join(root, 'splits.npz'), 
        splits=None)

    train_dataset = Subset(all_dataset, idx_train)
    val_dataset   = Subset(all_dataset, idx_val)
    test_dataset  = Subset(all_dataset, idx_test)

    return train_dataset, val_dataset, test_dataset
