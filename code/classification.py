import argparse
import os.path as osp
from os import walk
import os
import numpy as np
import torch
from torch_geometric.data import Dataset, download_url, Data, collate
from torch_geometric.utils import sparse
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import re
from tqdm import tqdm


DEVICE = 'cuda'


def get_args():
    parser = argparse.ArgumentParser(description="Classify by using GNN.")

    parser.add_argument("--features_dir_path", default='')
    parser.add_argument("--graph_pairs_path", default='')

    args, unknown = parser.parse_known_args()

    return args


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):

        self.classes_dict = {"HP": 0, "NCM": 1, "SSL": 2, "TA": 3}

        super().__init__(root, transform, pre_transform, pre_filter)

    def walker(self):
        paths = list()
        for (dirpath, dirnames, filenames) in walk(self.root):
            for j in filenames:
                paths.append(dirpath + '/' + j)
        return paths

    @property
    def size(self):
        return 8

    @property
    def num_node_features(self):
        return 512

    @property
    def num_classes(self):
        return 4

    @property
    def raw_file_names(self):
        return self.walker()

    @property
    def processed_file_names(self):
        return [f'data_{idx}.pt' for idx in range(self.size)]

    # def process(self):

    #     # Create a regex pattern for files with .pt extension
    #     pt_pattern = re.compile(r".*\.pt$")

    #     # Create a list to store file paths
    #     pt_file_paths = []

    #     # Use os.walk to traverse the directory recursively
    #     for root, dirs, files in os.walk(self.root):
    #         for file in files:
    #             # Check if the file matches the regex pattern
    #             if pt_pattern.match(file):
    #                 # Get the full path of the file and append it to the list
    #                 file_path = os.path.join(root, file)
    #                 pt_file_paths.append(file_path)

    #     idx = 0
    #     temp = []
    #     prev_slide_name = pt_file_paths[0].split("/")[-2]
    #     for raw_path in tqdm(pt_file_paths):
    #         adj_path = raw_path

    #         cls = raw_path.split("/")[-3]
    #         feature_path = raw_path.replace("output_graph", "output_cell_features").replace(
    #             cls+'/'+cls, cls+'/'+cls+'/'+cls)
    #         patch_name = feature_path.split("/")[-1].split('.')[0]
    #         slide_name = feature_path.split("/")[-2]
    #         feature_path = "/".join(feature_path.split("/")
    #                                 [:-1]) + '/raw/' + patch_name + '-features.pt'

    #         if not os.path.exists(feature_path):
    #             continue

    #         adj = torch.load(raw_path)
    #         # adj = torch.load(raw_path).to_sparse(layout=torch.sparse_coo)
    #         # # more efficient way to do this???????????
    #         # adj = sparse.to_edge_index(adj)
    #         if len(adj) == 0:
    #             continue
    #         edge_index = adj.T  # adj.indices()
    #         x = torch.Tensor(torch.load(feature_path))
    #         # print(x)
    #         y = self.classes_dict[cls]
    #         data = Data(x=x, edge_index=edge_index, y=y)

    #         if self.pre_filter is not None and not self.pre_filter(data):
    #             continue

    #         if self.pre_transform is not None:
    #             data = self.pre_transform(data)

    #         if prev_slide_name != slide_name:
    #             new_data, _, _ = collate.collate(Data, temp)
    #             torch.save(new_data, osp.join(self.processed_dir, f'data_{idx}.pt'))
    #             idx += 1
    #             temp = []
    #             prev_slide_name = slide_name

    #         temp.append(data)

    #     # TODO

    #     # pattern = re.compile(r'/([^/]+)/raw')

    #     # idx = 0
    #     # for raw_path in self.raw_paths:
    #     #     adj_path = raw_path
    #     #     print(self.raw_paths)
    #     #     feature_path = raw_path.replace("output_graph", "output_cell_features")[
    #     #         :-3] + '-features.pt'
    #     #     adj = torch.load(raw_path).to_sparse(layout=torch.sparse_coo)
    #     #     # # more efficient way to do this???????????
    #     #     # adj = sparse.to_edge_index(adj)
    #     #     edge_index = adj[0]  # adj.indices()
    #     #     x = torch.Tensor(torch.load(feature_path))
    #     #     y = self.classes_dict[pattern.search(adj_path).group(1)]
    #     #     data = Data(x=x, edge_index=edge_index, y=y)

    #     #     if self.pre_filter is not None and not self.pre_filter(data):
    #     #         continue

    #     #     if self.pre_transform is not None:
    #     #         data = self.pre_transform(data)

    #     #     torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
    #     #     idx += 1

    #     # # TODO: c64reate super graphs from processed data (after finilazing folder structure)

    def len(self):
        return int(len(os.listdir(self.processed_dir)))

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def train(model, criterion, optimizer, train_loader):
    model.train()

    # Iterate in batches over the training dataset.
    for data in tqdm(train_loader):
        # Perform a single forward pass.
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def evaluate(model, loader, mode='train'):
    model.eval()

    correct = 0
    preds = set()
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        preds.update([int(p) for p in pred])
        # Check against ground-truth labels.
        correct += int((pred == data.y).sum())

    print("model predicted classes in %s mode:" % (mode), list(preds))

    # Derive ratio of correct predictions.
    return correct / len(loader.dataset)


if __name__ == '__main__':
    args = get_args()
    dataset = MyOwnDataset(
        root=args.graph_pairs_path)

    torch.manual_seed(51)
    dataset = dataset.shuffle()

    # train_dataset = dataset[0::2]
    train_dataset = dataset[:120]
    # test_dataset = dataset[1::2]
    test_dataset = dataset[120:150]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4,
                             shuffle=True, num_workers=4, pin_memory=True)
    # for step, data in enumerate(train_loader):
    #     print(f'Step {step + 1}:')
    #     print('=======')
    #     print(f'Number of graphs in the current batch: {data.num_graphs}')
    #     print()

    model = GCN(hidden_channels=64)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 31):
        print(f'Epoch: {epoch:03d}')
        train(model, criterion, optimizer, train_loader)
        if epoch % 5 == 0:
            train_acc = evaluate(model, train_loader, "train")
            test_acc = evaluate(model, test_loader, "test")
            print(
                f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}\n################################################')
