import os
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from collections import defaultdict
import argparse

from utils import *


VISUALIZE = False


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract features of each cell based on extracted cell images.")

    parser.add_argument("--features_output_dir", default='')
    parser.add_argument("--cell_image_patches_dir", default='')
    parser.add_argument("--class_name", default="")

    args, unknown = parser.parse_known_args()

    return args


def init_directories(args):
    # Create a folder for each class in nuclei features directory
    os.makedirs(args.features_output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.features_output_dir, 'raw'), exist_ok=True)


if __name__ == '__main__':
    torch.manual_seed(0)

    args = get_parser()

    # Create dataset and dataloader
    dataset = InferenceDataset(
        path=args.cell_image_patches_dir, image_size=(224, 224))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
                            pin_memory=True)

    # Extract features
    vectors, patch_names = extract_features(dataloader, len(dataset))
    result_dict = defaultdict(lambda: torch.tensor([]))

    for key, value in zip(patch_names, vectors):
        result_dict[key] = torch.cat([result_dict[key], torch.tensor([value])])

    init_directories(args, args.class_name)

    # Save features
    for patch_name, features in result_dict.items():
        if VISUALIZE:
            pca = PCA()
            Xt = pca.fit_transform(features)
            plot = plt.scatter(Xt[:, 0], Xt[:, 1])
            plt.legend(handles=plot.legend_elements()[0])
            plt.savefig(patch_name + '.png')
            plt.close()

        patch_features_dir = os.path.join(args.features_output_dir, 'raw')
        torch.save(features, os.path.join(
            patch_features_dir, f'{patch_name}-features.pt'))
