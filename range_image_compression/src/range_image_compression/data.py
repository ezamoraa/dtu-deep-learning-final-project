import os
import glob
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T

class LidarCompressionDataset(Dataset):
    def __init__(self, input_dir, crop_size, augment=True):
        super().__init__()
        self.input_dir = input_dir
        self.crop_size = crop_size
        self.augment = augment
        self.file_list = glob.glob(os.path.join(input_dir, "*.png"))
        assert len(self.file_list) > 0, "No files found in {}".format(input_dir)
        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.RandomCrop(self.crop_size, pad_if_needed=augment, padding_mode='reflect'),
        ])

    def _get_image(self, index):
        file_path = self.file_list[index]
        image = torch.from_numpy(np.array(Image.open(file_path), dtype=np.int16))
        return image

    def __getitem__(self, index):
        image = self._get_image(index)
        image = torch.unsqueeze(self.transform(image), 0)
        return image

    def __len__(self):
        return len(self.file_list)