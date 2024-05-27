from torch.utils.data import DataLoader, Dataset
from ball.utils import transform
import os
import cv2
import numpy as np
import torch


class BallDataset(Dataset):
    def __init__(self, mode):
        super().__init__()
        self.root_path = "data/ball"
        assert mode in ["train", "val"], "incorrect mode"

        self.mode_path = os.path.join(self.root_path, f"{mode}")
        self.height = 360
        self.width = 640
        self.mode = mode

    def __len__(self):
        files = [file for file in os.listdir(os.path.join(self.mode_path, "images"))]
        return len(files) - 2

    def __getitem__(self, index):
        # index from 2 to len -1
        if self.mode == "val":
            index = index + 15000

        index += 2

        try:
            path1 = f"data/ball/{self.mode}/images/{index-1}.jpg"
            path2 = f"data/ball/{self.mode}/images/{index}.jpg"
            path3 = f"data/ball/{self.mode}/images/{index+1}.jpg"

            path_gt = f"data/ball/{self.mode}/labels/{index}.jpg"

            images = self.get_input(path1, path2, path3)
            gt = self.get_output(path_gt)

            return images, gt

        except Exception as e:
            return f"Error: {e}"

    def get_input(self, path1, path2, path3):
        image1 = cv2.imread(path1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        image2 = cv2.imread(path2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        image3 = cv2.imread(path3)
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

        image1 = transform(image1)
        image2 = transform(image2)
        image3 = transform(image3)

        images = np.concatenate((image1, image2, image3), axis=0)
        return torch.tensor(images)

    def get_output(self, path_gt):
        gt = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (640, 360))
        return torch.tensor(gt, dtype=torch.long).reshape(-1)


class GetLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def load_train_val(self):
        print("Loading train_val data...")
        train_data = BallDataset(mode="train")
        train_loader = DataLoader(
            train_data, self.batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        val_data = BallDataset(mode="val")
        val_loader = DataLoader(
            val_data, self.batch_size, shuffle=False, num_workers=4, drop_last=False
        )
        print("Loading completely...")
        return train_loader, val_loader


if __name__ == "__main__":
    data_loader = GetLoader(batch_size=32)
    train_loader, val_loader = data_loader.load_train_val()
    for images, gts in val_loader:
        print(images.dtype, gts.dtype)
