import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from augmentations import *


class WHSLoader(Dataset):
    def __init__(
        self, mode: str, classes: int, crop_dim: tuple, dataset_path: str, transform: bool
    ):
        self.mode = mode
        self.classes = classes
        self.crop_dim = crop_dim
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self):
        if self.mode == "train":
            return len(os.listdir(self.dataset_path+self.mode+"/imagesTr"))
        elif self.mode == "test":
            return len(os.listdir(self.dataset_path+self.mode+"/imagesTs"))

    def __getitem__(self, index: int):
        index = index+1
        if(index<10):
            index = "0"+str(index)
        if self.mode == "train":
            img_path = self.dataset_path+self.mode + \
                "/imagesTr/"+"mr_train_10"+str(index)+"_image"+".nii.gz"
        elif self.mode == "test":
            img_path = self.dataset_path+self.mode + \
                "/imagesTs/"+str(index)+".nii.gz"
        
        img = self.load_image(img_path)
        img = self.resize_image(img, self.crop_dim, mode="symmetric")
        img = self.normalize(img)

        if self.mode == "train":
            mask_path = self.dataset_path+self.mode + \
                "/labelsTr/"+"mr_train_10"+str(index)+"_label"+".nii.gz"
        elif self.mode == "test":
            mask_path = self.dataset_path+self.mode + \
                "/labelsTs/"+str(index)+".nii.gz"
        mask = self.load_image(mask_path)
        mask = self.resize_image(mask, self.crop_dim, mode="symmetric")
        
        transform = RandomFlip(p=0.5)
        img, mask = transform(img, mask)
        mask = self.preprocessing(mask)
        img = np.expand_dims(img, axis=0)
        return img, mask

    def preprocessing(self, mask: np.ndarray):
        mask_LV = mask.copy() # Left Ventricle
        mask_LV[mask_LV==500]=1
        mask_LV[mask_LV==600]=0
        mask_LV[mask_LV==420]=0
        mask_LV[mask_LV==550]=0
        mask_LV[mask_LV==205]=0
        mask_LV[mask_LV==820]=0
        mask_LV[mask_LV==850]=0

        mask_RV = mask.copy() # Right Ventricle
        mask_RV[mask_RV==500]=0
        mask_RV[mask_RV==600]=1
        mask_RV[mask_RV==420]=0
        mask_RV[mask_RV==550]=0
        mask_RV[mask_RV==205]=0
        mask_RV[mask_RV==820]=0
        mask_RV[mask_RV==850]=0

        mask_LA = mask.copy() # Left Atrium
        mask_LA[mask_LA==500]=0
        mask_LA[mask_LA==600]=0
        mask_LA[mask_LA==420]=1
        mask_LA[mask_LA==550]=0
        mask_LA[mask_LA==205]=0
        mask_LA[mask_LA==820]=0
        mask_LA[mask_LA==850]=0

        mask_RA = mask.copy() # Right Atrium
        mask_RA[mask_RA==500]=0
        mask_RA[mask_RA==600]=0
        mask_RA[mask_RA==420]=0
        mask_RA[mask_RA==550]=1
        mask_RA[mask_RA==205]=0
        mask_RA[mask_RA==820]=0
        mask_RA[mask_RA==850]=0


        mask_MC = mask.copy() # Myocardium
        mask_MC[mask_MC==500]=0
        mask_MC[mask_MC==600]=0
        mask_MC[mask_MC==420]=0
        mask_MC[mask_MC==550]=0
        mask_MC[mask_MC==205]=1
        mask_MC[mask_MC==820]=0
        mask_MC[mask_MC==850]=0


        mask_AA = mask.copy() # Ascending Aorta
        mask_AA[mask_AA==500]=0
        mask_AA[mask_AA==600]=0
        mask_AA[mask_AA==420]=0
        mask_AA[mask_AA==550]=0
        mask_AA[mask_AA==205]=0
        mask_AA[mask_AA==820]=1
        mask_AA[mask_AA==850]=0


        mask_PA = mask.copy() # Pulmonary Artery
        mask_PA[mask_PA==500]=0
        mask_PA[mask_PA==600]=0
        mask_PA[mask_PA==420]=0
        mask_PA[mask_PA==550]=0
        mask_PA[mask_PA==205]=0
        mask_PA[mask_PA==820]=0
        mask_PA[mask_PA==850]=1

        mask = np.stack([mask_LV, mask_RV, mask_LA, mask_RA, mask_MC, mask_AA, mask_PA])
        return mask

    def load_image(self, file_path: str):
        img = sitk.ReadImage(file_path)
        img = sitk.GetArrayFromImage(img)
        return img

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize_image(self, image: np.ndarray, img_size: tuple, **kwargs):
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
                to_padding[i][1] = img_size[i] - \
                    image.shape[i] - to_padding[i][0]
            else:
                from_indices[i][0] = int(
                    np.floor((image.shape[i] - img_size[i]) / 2.0))
                from_indices[i][1] = from_indices[i][0] + img_size[i]

            slicer[i] = slice(from_indices[i][0], from_indices[i][1])

        return np.pad(image[slicer], to_padding, **kwargs)

if __name__ == "__main__":
    dataset = WHSLoader(mode="train",classes=7, crop_dim=(32, 256, 256), dataset_path="../mr_train/",transform=True)
    for i in range(len(dataset)):
        img, mask = dataset[i]
        print(img.shape)
        print(mask.shape)
        fig,ax = plt.subplots(3,3)
        ax[0,0].imshow(img[0][20],cmap="bone")
        ax[0,1].imshow(mask[0][20],cmap="bone")
        ax[0,2].imshow(mask[1][20],cmap="bone")
        ax[1,0].imshow(mask[2][20],cmap="bone")
        ax[1,1].imshow(mask[3][20],cmap="bone")
        ax[1,2].imshow(mask[4][20],cmap="bone")
        ax[2,0].imshow(mask[5][20],cmap="bone")
        ax[2,1].imshow(mask[6][20],cmap="bone")
        
        plt.show()
        break
        
        
       