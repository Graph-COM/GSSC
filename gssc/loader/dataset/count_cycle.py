"""
modified from https://github.com/GraphPKU/I2GNN/blob/master/data_processing.py
"""

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
import os
from torch_geometric.data import Data, InMemoryDataset
import scipy.io as scio


class CountCycle(InMemoryDataset):
    def __init__(
        self,
        dataname="count_cycle",
        root="dataset",
        processed_name="processed",
        split="train",
        yidx: int = 0,
        ymean: float = 0,
        ystd: float = 1,
        ymean_log: float = 0,
        ystd_log: float = 1,
        replace=False,
        transform=None,
    ):
        self.root = root
        self.dataname = dataname
        self.raw = os.path.join(root, dataname)
        self.processed = os.path.join(root, dataname, processed_name)
        super(CountCycle, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)
        split_id = 0 if split == "train" else 1 if split == "val" else 2
        data, slices = torch.load(self.processed_paths[split_id])

        data.log_y = torch.log10(data.y + 1.0)
        data.log_y = (data.log_y[:, [yidx]] - ymean_log) / ystd_log
        data.log_y = data.log_y.reshape(-1)

        data.y = (data.y[:, [yidx]] - ymean) / ystd
        data.y = data.y.reshape(-1)

        if replace:
            data.y = data.log_y

        self.data, self.slices = data, slices
        self.mean, self.std, self.log_mean, self.log_std = ymean, ystd, ymean_log, ystd_log
        # ((10**((data.log_y * ystd_log) + ymean_log) - 1) - ymean) / ystd

    @property
    def raw_dir(self):
        name = "raw"
        return os.path.join(self.root, self.dataname, name)

    @property
    def processed_dir(self):
        return self.processed

    @property
    def raw_file_names(self):
        names = ["data"]
        return ["{}.mat".format(name) for name in names]

    @property
    def processed_file_names(self):
        return ["data_tr.pt", "data_val.pt", "data_te.pt"]

    def adj2data(self, A, y):
        # x: (n, d), A: (e, n, n)
        # begin, end = np.where(np.sum(A, axis=0) == 1.)
        begin, end = np.where(A == 1.0)
        edge_index = torch.tensor(np.array([begin, end]))
        num_nodes = A.shape[0]
        if y.ndim == 1:
            y = y.reshape([1, -1])
        x = torch.ones((num_nodes, 1), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=torch.tensor(y), num_nodes=torch.tensor([num_nodes]))

    def process(self):
        # process npy data into pyg.Data
        print("Processing data from " + self.raw_dir + "...")
        raw_data = scio.loadmat(self.raw_paths[0])
        if raw_data["F"].shape[0] == 1:
            data_list_all = [
                [self.adj2data(raw_data["A"][0][i], raw_data["F"][0][i]) for i in idx]
                for idx in [raw_data["train_idx"][0], raw_data["val_idx"][0], raw_data["test_idx"][0]]
            ]
        else:
            data_list_all = [
                [self.adj2data(A, y) for A, y in zip(raw_data["A"][0][idx][0], raw_data["F"][idx][0])]
                for idx in [raw_data["train_idx"], raw_data["val_idx"], raw_data["test_idx"]]
            ]
        for save_path, data_list in zip(self.processed_paths, data_list_all):
            print("pre-transforming for data at" + save_path)
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                temp = []
                for i, data in enumerate(data_list):
                    if i % 100 == 0:
                        print("Pre-processing %d/%d" % (i, len(data_list)))
                    temp.append(self.pre_transform(data))
                data_list = temp
                # data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            torch.save((data, slices), save_path)
