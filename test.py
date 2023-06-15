import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt



img01 = sitk.ReadImage("../iSeg/img/subject-10-T1-00.nii.gz")
img01 = sitk.GetArrayFromImage(img01)

img02 = sitk.ReadImage("../iSeg/img/subject-10-T1-01.nii.gz")
img02 = sitk.GetArrayFromImage(img02)

img03 = sitk.ReadImage("../iSeg/img/subject-10-T1-02.nii.gz")
img03 = sitk.GetArrayFromImage(img03)

img04 = sitk.ReadImage("../iSeg/img/subject-10-T1-03.nii.gz")
img04 = sitk.GetArrayFromImage(img04)

img05 = sitk.ReadImage("../iSeg/img/subject-10-T1-04.nii.gz")
img05 = sitk.GetArrayFromImage(img05)

img06 = sitk.ReadImage("../iSeg/img/subject-10-T1-05.nii.gz")
img06 = sitk.GetArrayFromImage(img06)

img07 = sitk.ReadImage("../iSeg/img/subject-10-T1-06.nii.gz")
img07 = sitk.GetArrayFromImage(img07)

img08 = sitk.ReadImage("../iSeg/img/subject-10-T1-07.nii.gz")
img08 = sitk.GetArrayFromImage(img08)


img11 = sitk.ReadImage("../iSeg/img/subject-10-T2-00.nii.gz")
img11 = sitk.GetArrayFromImage(img11)

img12 = sitk.ReadImage("../iSeg/img/subject-10-T2-01.nii.gz")
img12 = sitk.GetArrayFromImage(img12)

img13 = sitk.ReadImage("../iSeg/img/subject-10-T2-02.nii.gz")
img13 = sitk.GetArrayFromImage(img13)

img14 = sitk.ReadImage("../iSeg/img/subject-10-T2-03.nii.gz")
img14 = sitk.GetArrayFromImage(img14)

img15 = sitk.ReadImage("../iSeg/img/subject-10-T2-04.nii.gz")
img15 = sitk.GetArrayFromImage(img15)

img16 = sitk.ReadImage("../iSeg/img/subject-10-T2-05.nii.gz")
img16 = sitk.GetArrayFromImage(img16)

img17 = sitk.ReadImage("../iSeg/img/subject-10-T2-06.nii.gz")
img17 = sitk.GetArrayFromImage(img17)

img18 = sitk.ReadImage("../iSeg/img/subject-10-T2-07.nii.gz")
img18 = sitk.GetArrayFromImage(img18)

img1 = np.stack([img01,img11])
img2 = np.stack([img02,img12])
img3 = np.stack([img03,img13])
img4 = np.stack([img04,img14])
img5 = np.stack([img05,img15])
img6 = np.stack([img06,img16])
img7 = np.stack([img07,img17])
img8 = np.stack([img08,img18])


# Save the image as nii format
sitk.WriteImage(sitk.GetImageFromArray(img1), "../iSeg/image/subject-10-00.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(img2), "../iSeg/image/subject-10-01.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(img3), "../iSeg/image/subject-10-02.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(img4), "../iSeg/image/subject-10-03.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(img5), "../iSeg/image/subject-10-04.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(img6), "../iSeg/image/subject-10-05.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(img7), "../iSeg/image/subject-10-06.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(img8), "../iSeg/image/subject-10-07.nii.gz")




print(img1.shape)
print(img2.shape)
print(img3.shape)
print(img4.shape)
print(img5.shape)
print(img6.shape)
print(img7.shape)
print(img8.shape)



#label = resize_image(label, (128, 128, 128))

'''print(img.shape)
print(label.shape)
plt.figure(1)
plt.imshow(img[80], cmap='bone')
plt.figure(2)
plt.imshow(label[80], cmap='bone')
plt.show()'''