from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class BreastSegmentationDataset(Dataset):
    def __init__(self, dataframe, image_dir, image_size=224):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_dir / row["Image_filename"]
        mask_path = self.image_dir / row["Mask_tumor_filename"]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.NEAREST
        )

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)

        mask = (mask > 0).astype(np.float32)

        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask