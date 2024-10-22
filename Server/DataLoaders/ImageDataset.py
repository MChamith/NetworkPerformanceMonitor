import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from .preprocessUtil import get_transformations

class ImageDataset(Dataset):

    def __init__(self, dataset, labels, transformations):
        super().__init__()
        self.dataset = dataset
        self.labels = labels
        self.transform = transforms.Compose(get_transformations(transformations))
        self.target_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index].astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        return image, label