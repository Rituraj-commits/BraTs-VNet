import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from augmentations import *
def resize_image(image, img_size=(64, 64, 64), **kwargs):
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

transform = RandomFlip(p=0.5)

img = sitk.ReadImage("../mr_train/train/imagesTr/mr_train_1011_image.nii.gz")
img = sitk.GetArrayFromImage(img)


#img = resize_image(img, (32, 256, 256))
label = sitk.ReadImage("../mr_train/train/labelsTr/mr_train_1011_label.nii.gz")
label = sitk.GetArrayFromImage(label)

img, label = transform(img, label)



#label = resize_image(label, (32, 256, 256))

print(img.shape)
print(label.shape)
plt.figure(1)
plt.imshow(img[70], cmap='bone')
plt.figure(2)
plt.imshow(label[70], cmap='bone')
plt.show()