import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from torch import nn
from torchvision import transforms
import timm
from tqdm import tqdm
import PIL.Image as Image


BATCH_SIZE = 32
DIMENSION = 64


class InferenceDataset(Dataset):
    def __init__(self, path, image_size, image_format='png'):
        self.path = path
        self.paths = list(Path(self.path).glob("*." + image_format))
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

        return image

    @staticmethod
    def get_transform(crop_size=(32, 32)):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose(
            [
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


def extract_features(dataloader, n, model_name='resnet18', use_mps=True):
    vectors = torch.zeros(n, DIMENSION)

    # Load pre-trained model
    model = timm.create_model(model_name, pretrained=True)

    device = torch.device("cpu")
    if use_mps:
        # related to mac
        device = torch.device("mps")

    # Set the model to evaluation mode
    model.eval()

    # model.global_pool.register_forward_hook(get_activation('global_pool'))
    # avg_pool = nn.AvgPool1d(8)

    # for i, tensors in enumerate(tqdm(dataloader)):
    #     tensors = tensors.to(mps_device)
    #     features = model(tensors)
    #     global_pool_output = activation['global_pool']
    #     features = avg_pool(global_pool_output)
    #     vectors[i*BATCH_SIZE: (i+1)*BATCH_SIZE, :] = features

    model.fc = nn.Linear(512, DIMENSION)
    model = model.to(device)

    
    for i, tensors in enumerate(tqdm(dataloader)):
        tensors = tensors.to(device)
        with torch.no_grad():
            features = model(tensors)
        print(features)
        vectors[i*BATCH_SIZE: (i+1)*BATCH_SIZE, :] = features.squeeze().cpu()

    

    # Convert the features to a vector representation
    vector_representation = vectors.squeeze().cpu().numpy()

    return vector_representation
