import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from augmentations import *
from random import randint


class ISEGLoader(Dataset):
    def __init__(self, mode: str, dataset_path: str):
        self.mode = mode
        self.dataset_path = dataset_path

    def __len__(self):
        if self.mode == "train":
            return 8
        elif self.mode == "test":
            return 2

    def __getitem__(self, index: int):
        index = index + 1
        idx = randint(0,7)
        if self.mode == "train":
            img_path = self.dataset_path + self.mode + "/image/subject-" + str(index) + "-0" + str(idx) + ".nii.gz"

        elif self.mode == "test":
            img_path = self.dataset_path + self.mode + "/image/subject-" + str(index) + "-0" + str(idx) + ".nii.gz"


        img = self.load_image(img_path)
        images = []
        img1 = self.normalize(img[0, :, :])  # T1
        images.append(img1)

        img2 = self.normalize(img[1, :, :])  # T2
        images.append(img2)

        img = np.stack(images)

        if self.mode == "train":
            img_path = self.dataset_path + self.mode + "/label/subject-"+str(index)+"-label-" + "0"+str(idx)+".nii.gz"
        elif self.mode == "test":
            img_path = self.dataset_path + self.mode + "/label/subject-"+str(index)+"-label-"+"0"+str(idx)+".nii.gz"

        mask = self.load_image(img_path)
        mask = self.preprocessing(mask)

        img = np.array(img, copy=True)
        mask = np.array(mask, copy=True)

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


'''def has_negative_strides(arr):
    return any(stride < 0 for stride in arr.strides)
# Visualize the data
if __name__ == "__main__":
    dataset = ISEGLoader(mode="train", dataset_path="../iSeg/")
    for i in range(len(dataset)):
        img, mask = dataset[i]
        print(has_negative_strides(img))
        print(has_negative_strides(mask))
        print(img.shape)
        print(mask.shape)
        fig,ax = plt.subplots(2,3)
        ax[0,0].imshow(img[0][80],cmap="bone")
        ax[0,1].imshow(img[1][80],cmap="bone")
        ax[0,2].imshow(mask[0][80],cmap="bone")
        ax[1,0].imshow(mask[1][80],cmap="bone")
        ax[1,1].imshow(mask[2][80],cmap="bone")
        plt.show()
        break'''
