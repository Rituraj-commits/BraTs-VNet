import numpy as np
import SimpleITK as sitk
import os
from torch.utils import data
from torch.utils.data import Dataset,DataLoader
import json


class BratsDataset(Dataset):
    def __init__(
        self, mode, classes=3, crop_dim=(64, 64, 64), dataset_path="./BraTs_data/"
    ):
        self.mode = mode
        self.classes = classes
        self.crop_dim = crop_dim
        self.dataset_path = dataset_path

    def __len__(self):
        if self.mode == "train":
            return len(os.listdir(self.dataset_path + self.mode + "/imagesTr")) - 11
        elif self.mode == "test":
            return len(os.listdir(self.dataset_path + self.mode + "/imagesTr"))

    def __getitem__(self, index):
        file = open("Task01_BrainTumour/dataset.json")
        data = json.load(file)
        img = []
        label = []
        img_test = []
        label_test = []
        for i in data["training"]:
            x = i["image"]
            y = i["label"]
            if x.startswith("./"):
                x = x[2:]
                img.append(x)
                img.sort()
            if y.startswith("./"):
                y = y[2:]
                label.append(y)
                label.sort()
        for j in data["test"]:
            x = j["image"]
            y = j["label"]
            if x.startswith("./"):
                x = x[2:]
                img_test.append(x)
                img_test.sort()
            if y.startswith("./"):
                y = y[2:]
                label_test.append(y)
                label_test.sort()
        file.close()

        if self.mode == "train":
            img_path = self.dataset_path + self.mode + "/" + img[index]
        elif self.mode == "test":
            img_path = self.dataset_path + self.mode + "/" + img_test[index]
        img = self.load_image(img_path)
        img1 = self.resize_image(img[0, :, :], self.crop_dim, mode="symmetric")  # Flair
        img2 = self.resize_image(img[1, :, :], self.crop_dim, mode="symmetric")  # T1w
        img3 = self.resize_image(img[2, :, :], self.crop_dim, mode="symmetric")  # t1gd
        img4 = self.resize_image(img[3, :, :], self.crop_dim, mode="symmetric")  # T2w
        img = np.stack((img1, img2, img3, img4), axis=0)
        img = self.normalize(img)

        if self.mode == "train":
            mask_path = self.dataset_path + self.mode + "/" + label[index]
        elif self.mode == "test":
            mask_path = self.dataset_path + self.mode + "/" + label_test[index]
        mask = self.load_image(mask_path)
        mask = self.resize_image(mask, self.crop_dim, mode="symmetric")
        mask = np.clip(mask, 0, 1)
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET]).astype(np.float32)

        return img, mask

    def load_image(self, file_path):
        img = sitk.ReadImage(file_path)
        img = sitk.GetArrayFromImage(img)
        return img

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize_image(self, image, img_size=(64, 64, 64), **kwargs):
        assert isinstance(image, (np.ndarray, np.generic))
        assert image.ndim - 1 == len(img_size) or image.ndim == len(
            img_size
        ), "Example size doesnt fit image size"

        rank = len(img_size)

        from_indices = [[0, image.shape[dim]] for dim in range(rank)]
        to_padding = [[0, 0] for dim in range(rank)]

        slicer = [slice(None)] * rank

        for i in range(rank):
            if image.shape[i] < img_size[i]:
                to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
                to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
            else:
                from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.0))
                from_indices[i][1] = from_indices[i][0] + img_size[i]

            slicer[i] = slice(from_indices[i][0], from_indices[i][1])

        return np.pad(image[slicer], to_padding, **kwargs)
