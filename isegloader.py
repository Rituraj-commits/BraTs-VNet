import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from augmentations import *


class ISEGLoader(Dataset):
    def __init__(self, mode: str, classes: int, dataset_path: str):
        self.mode = mode
        self.classes = classes
        self.dataset_path = dataset_path

    def __len__(self):
        if self.mode == "train":
            return len(os.listdir(self.dataset_path+self.mode+"/img"))
        elif self.mode == "test":
            return len(os.listdir(self.dataset_path+self.mode+"/img"))

    def __getitem__(self, index: int):
        index = index + 1
        if self.mode == "train":
            img_path = self.dataset_path+self.mode + \
                "/img/"+"subject-"+str(index)+".nii.gz"
        elif self.mode == "test":
            img_path = self.dataset_path+self.mode + \
                "/img/"+"subject-"+str(index)+".nii.gz"

        img = self.load_image(img_path)
        img = self.normalize(img)

        if self.mode == "train":
            img_path = self.dataset_path+self.mode + \
                "/label/"+"subject-"+str(index)+"-label.nii.gz"
        elif self.mode == "test":
            img_path = self.dataset_path+self.mode + \
                "/label/"+"subject-"+str(index)+"-label.nii.gz"

        mask = self.load_image(img_path)
        mask = self.preprocessing(mask)

        transform = RandomFlip(p=0.5)
        img, mask = transform(img, mask)

        return img, mask

    def preprocessing(self, mask: np.ndarray):
        mask_WM = mask.copy()  # White Matter
        mask_WM[mask_WM == 250] = 1
        mask_WM[mask_WM == 150] = 0
        mask_WM[mask_WM == 10] = 0

        mask_GM = mask.copy()  # Gray Matter
        mask_GM[mask_GM == 250] = 0
        mask_GM[mask_GM == 150] = 1
        mask_GM[mask_GM == 10] = 0

        mask_CSF = mask.copy()  # Cerbrospinal Fluid
        mask_CSF[mask_CSF == 250] = 0
        mask_CSF[mask_CSF == 150] = 0
        mask_CSF[mask_CSF == 10] = 1

        mask = np.stack([mask_WM, mask_GM, mask_CSF]).astype(np.float32)
        return mask

    def load_image(self, file_path: str):
        img = sitk.ReadImage(file_path)
        img = sitk.GetArrayFromImage(img)
        return img

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

# Visualize the data
'''if __name__ == "__main__":
    dataset = ISEGLoader(mode="train", classes=2, dataset_path="../iSeg-2017/")
    for i in range(len(dataset)):
        img, mask = dataset[i]
        print(img.shape)
        print(mask.shape)
        fig,ax = plt.subplots(2,3)
        ax[0,0].imshow(img[0][40],cmap="bone")
        ax[0,1].imshow(img[1][40],cmap="bone")
        ax[0,2].imshow(mask[0][40],cmap="bone")
        ax[1,0].imshow(mask[1][40],cmap="bone")
        ax[1,1].imshow(mask[2][40],cmap="bone")
        plt.show()
        break'''
