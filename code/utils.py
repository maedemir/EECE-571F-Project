import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from torch import nn
from torchvision import transforms
import timm
from tqdm import tqdm
import PIL.Image as Image
import numpy as np
import torchvision.transforms.functional as function
import json
import os


BATCH_SIZE = 128
DIMENSION = 64


def open_json(json_path):
    with open(json_path) as f:
        return json.load(f)
    
    
def get_class_name(file_name):
    class_name = file_name.split('-')[-1].split('_')[0]
    return class_name

def get_centroids(args):
    centroids_per_file = {}
    cls_json_dir_path = args.json_dir_path
    for json_file_name in os.listdir(cls_json_dir_path):
        json_path = os.path.join(cls_json_dir_path, json_file_name)
        json_content = open_json(json_path)

        centroid_values = {}
        for key, value in json_content.get("nuc", {}).items():
            centroid = value.get("centroid")
            if centroid is not None:
                centroid_values[key] = centroid

        centroids_per_file[json_file_name] = centroid_values
    return centroids_per_file


def get_bboxes(args):
    bbox_per_file = {}
    cls_json_dir_path = args.json_dir
    for json_file_name in os.listdir(cls_json_dir_path):
        json_path = os.path.join(cls_json_dir_path, json_file_name)
        json_content = open_json(json_path)

        bbox_values = {}
        for key, value in json_content.get("nuc", {}).items():
            bbox = value.get("bbox")
            if bbox is not None:
                bbox_values[key] = bbox

        bbox_per_file[json_file_name] = bbox_values
    return bbox_per_file


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return function.pad(image, padding, 0, 'constant')


class InferenceDataset(Dataset):
    def __init__(self, path, image_size, image_format='png'):
        self.path = path
        # self.paths = list(Path(self.path).glob("*." + image_format))
        self.paths = list(Path(self.path).glob("*/*." + image_format))
        self.paths.sort()
        self.transform = self.get_transform(image_size)
        self.image_size = image_size[0]

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def load_image(image_path):
        return Image.open(image_path).convert('RGB')

    def __getitem__(self, index):
        image_path = str(self.paths[index])
        image = self.load_image(image_path)
        image = self.transform(image)

        patch_name = '-'.join(image_path.split('/')[-1].split('-')[:-1])
        return image, patch_name

    @staticmethod
    def get_transform(crop_size=(32, 32)):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose(
            [
                # SquarePad(),
                transforms.Resize(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

        return transform


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# vit Small: vit_small_patch16_224
def extract_features(dataloader, n, model_name='resnet18', use_cuda=True):
    vectors = torch.zeros(n, DIMENSION)
    patch_names = []

    # Load pre-trained model
    model = timm.create_model(model_name, pretrained=True)

    device = torch.device("cpu")
    if use_cuda:
        # related to mac
        device = torch.device("cuda")

    # Set the model to evaluation mode
    model.eval()

    # #### ViT Small
    # model = model.to(device)
    # model.norm.register_forward_hook(get_activation('norm'))
    # for i, (tensors, patch_name_tensor) in enumerate(tqdm(dataloader)):
    #     tensors = tensors.to(device)
    #     with torch.no_grad():
    #         model(tensors)
    #         features = activation['norm'][:, -1, :]
    #     vectors[i*BATCH_SIZE: (i+1)*BATCH_SIZE, :] = features.squeeze().cpu()
    #     patch_names += patch_name_tensor
        
    
    #### ResNet18
    model.fc = nn.Linear(512, DIMENSION)
    model = model.to(device)
    model.norm.register_forward_hook(get_activation('fc'))
    
    for i, (tensors, patch_name_tensor) in enumerate(tqdm(dataloader)):
        tensors = tensors.to(device)
        with torch.no_grad():
            model(tensors)
            features = activation['fc']
        vectors[i*BATCH_SIZE: (i+1)*BATCH_SIZE, :] = features.squeeze().cpu()
        patch_names += patch_name_tensor



    # Convert the features to a vector representation

    return vectors.numpy(), patch_names
