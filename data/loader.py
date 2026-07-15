import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.data_utils import chipImage


class ChipDataset(Dataset):

    def __init__(self, img_path, mask_path=None):
        if mask_path is not None:
            self.chips, self.masks, self.meta = chipImage(img_path=img_path, mask_path=mask_path)
        else:
            self.chips, self.meta = chipImage(img_path=img_path, mask_path=mask_path)
            self.masks = None

        print(f"[Init] ChipDataset initialized with {len(self.chips)} chips.")

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, index):
        image = self._normalize(self.chips[index])
        if self.masks is not None:
            label = torch.from_numpy(self.masks[index].astype(np.float32)).float()
            return image, label
        else:
            return image, index 

    @property
    def get_meta(self):
        return self.meta

    def _normalize(self, x):
        x = x.astype(np.float32)
        x_min = x.min(axis=(1, 2), keepdims=True) 
        x_max = x.max(axis=(1, 2), keepdims=True)
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)
        return torch.from_numpy(x_norm).float()


def createdataset(img_path, mask_path=None):
    dataset = ChipDataset(img_path, mask_path=mask_path)
    loader = DataLoader(dataset, batch_size=1, num_workers=4,
                        pin_memory=False, drop_last=False, shuffle=False)
    return loader
