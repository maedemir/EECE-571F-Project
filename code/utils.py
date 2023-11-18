import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from torch import nn
from torchvision import transforms
import timm
from tqdm import tqdm
import PIL.Image as Image


BATCH_SIZE = 64
DIMENSION = 384


class InferenceDataset(Dataset):
    def __init__(self, path, image_size, image_format='png'):
        self.path = path
        # self.paths = list(Path(self.path).glob("*." + image_format))
        self.paths = list(Path(self.path).glob("*/*/*." + image_format))
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
def extract_features(dataloader, n, model_name='vit_small_patch16_224', use_mps=True):
    vectors = torch.zeros(n, DIMENSION)
    patch_names = []

    # Load pre-trained model
    model = timm.create_model(model_name, pretrained=True)

    device = torch.device("cpu")
    if use_mps:
        # related to mac
        device = torch.device("mps")

    # Set the model to evaluation mode
    model.eval()

    #### ViT Small
    model = model.to(device)
    model.norm.register_forward_hook(get_activation('norm'))
    for i, (tensors, patch_name_tensor) in enumerate(tqdm(dataloader)):
        tensors = tensors.to(device)
        with torch.no_grad():
            model(tensors)
            features = activation['norm'][:, -1, :]
        vectors[i*BATCH_SIZE: (i+1)*BATCH_SIZE, :] = features.squeeze().cpu()
        patch_names += patch_name_tensor
        
    
    # #### ResNet18
    # model.fc = nn.Linear(512, DIMENSION)
    # model = model.to(device)
    
    # for i, (tensors, patch_name_tensor) in enumerate(tqdm(dataloader)):
    #     tensors = tensors.to(device)
    #     with torch.no_grad():
    #         features = model(tensors)
    #     vectors[i*BATCH_SIZE: (i+1)*BATCH_SIZE, :] = features.squeeze().cpu()
    #     patch_names += patch_name_tensor



    # Convert the features to a vector representation

    return vectors.numpy(), patch_names
