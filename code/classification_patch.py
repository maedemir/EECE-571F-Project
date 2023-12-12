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
from torch_geometric.nn import GCNConv, TransformerConv, SuperGATConv, GATv2Conv, SGConv, SAGEConv
from torch_geometric.nn import global_mean_pool
import re
from tqdm import tqdm
from torch_geometric.nn import LayerNorm
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
import random
from collections import defaultdict
from torch.utils.data import Subset





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
    def num_node_features(self):
        return 64

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
        file_name = os.listdir(self.processed_dir)[idx]
        slide_name = file_name.split('_')[0]
        data = torch.load(osp.join(self.processed_dir, file_name))
        return data


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_node_features, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
        # self.conv3 = GATv2Conv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.conv1 = TransformerConv(dataset.num_node_features, hidden_channels)
        # self.conv2 = TransformerConv(hidden_channels, hidden_channels)
        # self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        self.layer_norm = LayerNorm(hidden_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.layer_norm(x)
        x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.layer_norm(x)
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin(x)

        return x
    
    
metrics = {}
def create_metrics(num_classes):  
    metrics['valid_accuracy'] = Accuracy(task="multiclass", num_classes=num_classes).to('cuda')
    metrics['valid_average_accuracy'] = Accuracy(task="multiclass", num_classes=num_classes, average='macro').to('cuda')
    metrics['valid_precision'] = Precision(task="multiclass", num_classes=num_classes, average='macro').to('cuda')
    metrics['valid_recall'] = Recall(task="multiclass", num_classes=num_classes, average='macro').to('cuda')
    metrics['valid_f1'] = F1Score(task="multiclass", num_classes=num_classes, average='macro').to('cuda')
    metrics['valid_auc'] = AUROC(task="multiclass", num_classes=num_classes)
    

def update_metrics(true, pred, pred_p):
    for key, metric in metrics.items():
        if key == 'valid_auc':
            metric.update(pred_p, true)
        else:
            metric.update(pred, true)
    

def compute_metrics():
    print('* valid accuracy =', metrics['valid_accuracy'].compute().item())
    print('* valid average accuracy =', metrics['valid_average_accuracy'].compute().item())
    print('* valid precision =', metrics['valid_precision'].compute().item())
    print('* valid recall =', metrics['valid_recall'].compute().item())
    print('* valid f1 =', metrics['valid_f1'].compute().item())
    print('* valid auc =', metrics['valid_auc'].compute().item())
    print('\n\n')



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

    create_metrics(num_classes=4)
    correct = 0
    preds = set()
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        update_metrics(data.y, pred, out)
        preds.update([int(p) for p in pred])
        # Check against ground-truth labels.
        correct += int((pred == data.y).sum())
    compute_metrics()

    print("model predicted classes in %s mode:" % (mode), list(preds))

    # Derive ratio of correct predictions.
    return correct / len(loader.dataset)


def prepare_splits(dataset):
    samples_map = defaultdict(lambda: defaultdict(list))

    for i, (file_name) in enumerate(os.listdir(dataset.processed_dir)):
        slide_name = file_name.split('_')[0]
        class_name = slide_name.split('-')[0]
        samples_map[class_name][slide_name].append(i)

    splits = [[[], []],
              [[], []],
              [[], []]]

    for cls, slide_dict in samples_map.items():
        slide_names = list(slide_dict.keys())
        n_slides = len(slide_names)
        fold_size = n_slides // 3
        
        for fold_i in range(3):
            start_idx = fold_i * fold_size
            end_idx = (fold_i + 1) * fold_size if fold_i < 2 else n_slides
            
            train_slides = slide_names[:start_idx] + slide_names[end_idx:]
            valid_slides = slide_names[start_idx:end_idx]
            
            train_patches = [slide_dict[x] for x in train_slides]
            train_patches = sum(train_patches, [])
            
            valid_patches = [slide_dict[x] for x in valid_slides]
            valid_patches = sum(valid_patches, [])
            
            splits[fold_i][0] += train_patches
            splits[fold_i][1] += valid_patches
    
    return splits

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)


    args = get_args()
    dataset = MyOwnDataset(root=args.graph_pairs_path)
    splits = prepare_splits(dataset)
    print("fold 1: training size =", len(splits[0][0]))
    print("fold 1: validation size =", len(splits[0][1]))
    print("fold 2: training size =", len(splits[1][0]))
    print("fold 2: validation size =", len(splits[1][1]))
    print("fold 3: training size =", len(splits[2][0]))
    print("fold 3: validation size =", len(splits[2][1]))

    n_data = len(dataset)
    
    test_acc_list = []

    # cross-validation
    for fold_i in range(3):
        train_idx = splits[fold_i][0]
        valid_idx = splits[fold_i][1]
        
        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        
        print('#' * 30 + " Fold: " + str(fold_i + 1) + " " + '#' * 30)
        
        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(valid_dataset)}')
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True, num_workers=2)
        
        
        model = GCN(hidden_channels=64)
        model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        best_acc_valid = 0
        best_model = None

        for epoch in range(0, 13):
            print(f'Epoch: {epoch:03d}')
            train(model, criterion, optimizer, train_loader)
            if epoch % 3 == 0:
                # train_acc = evaluate(model, train_loader, "train")
                test_acc = evaluate(model, valid_loader, "test")
                print('test_acc:', test_acc)
                
                if test_acc > best_acc_valid:
                    best_acc_valid = test_acc
                    best_model = model
                    test_acc_list.append(test_acc)

                
                # print(
                #     f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}\n################################################')
    
    print('Mean ACC on Valid =', sum(test_acc_list) / len(test_acc_list))

   
