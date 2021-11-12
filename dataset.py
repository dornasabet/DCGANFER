from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

import config


class Dataset_fer(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.images = os.listdir(root)
        self.length_dataset = len(self.images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        img = self.images[index % self.length_dataset]

        img_path = os.path.join(self.root, img)

        if config.IMG_CHANNEL == 3:
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)
        return img
