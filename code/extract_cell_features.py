import numpy as np
import os
import shutil
import json
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from collections import defaultdict

from utils import *


# torch.manual_seed(0)


JSON_DIR_PATH = 'data/json'
GRAPH_OUTPUT_DIR = 'data/graph_adj_matrix'
FEATURES_OUTPUT_DIR = 'data/nuclei_features'
IMAGE_PATH = 'data/imgs'
CELL_IMAGE_PATCHES_DIR = 'data/extracted_cells'
CLASSES = ['HP', 'NCM', 'SSL', 'TA']
ENABLE_PCA = False


def get_class_name(file_name):
    class_name = file_name.split('-')[-1].split('_')[0]
    return class_name


def init_directories():
    for class_name in CLASSES:
        os.path.join(CELL_IMAGE_PATCHES_DIR, class_name)

        # Create a folder for each class in nuclei features directory
        os.makedirs(os.path.join(FEATURES_OUTPUT_DIR,
                    class_name), exist_ok=True)

        patch_features_dir = os.path.join(
            FEATURES_OUTPUT_DIR, class_name, 'raw')
        os.makedirs(patch_features_dir, exist_ok=True)


if __name__ == '__main__':
    dataset = InferenceDataset(
        path=CELL_IMAGE_PATCHES_DIR, image_size=(224, 224))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
                            pin_memory=True)

    vectors, patch_names = extract_features(dataloader, len(dataset))
    result_dict = defaultdict(lambda: torch.tensor([]))

    for key, value in zip(patch_names, vectors):
        result_dict[key] = torch.cat([result_dict[key], torch.tensor([value])])

    init_directories()

    for patch_name, features in result_dict.items():
        if ENABLE_PCA:
            pca = PCA()
            Xt = pca.fit_transform(features)
            plot = plt.scatter(Xt[:, 0], Xt[:, 1])
            plt.legend(handles=plot.legend_elements()[0])
            plt.savefig(patch_name + '.png')
            plt.close()

        class_name = get_class_name(patch_name)
        patch_features_dir = os.path.join(
            FEATURES_OUTPUT_DIR, class_name, 'raw')
        torch.save(features, os.path.join(
            patch_features_dir, f'{patch_name}-features.pt'))
