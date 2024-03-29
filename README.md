# 3D Automatic Brain Tumor Segmentation using Fully Convolutional Networks

This project is an approach to detect brain tumours using BraTS 2016,2017 dataset.

## Description

[BraTS](http://medicaldecathlon.com/) is a dataset which provides multimodal 3D brain MRIs annotated by experts. Each Magnetic Resonance Imaging(MRI) scan consists of 4 different modalities(Flair,T1w,t1gd,T2w).
Expert annotations are provided in the form of segmentation masks to detect 3 classes of tumour - edema(ED),enhancing tumour(ET),necrotic and non-enhancing tumour(NET/NCR). The dataset is challenging in terms of the complex and heterogeneously-located targets.
We use Volumetric Network(V-Net) which is a 3D Fully Convolutional Network(FCN) for segmentation of 3D medical images. We use Dice Loss as the objective function for the present scenario. Future implementation will include Hausdorff Loss for better boundary segmentations.

<div align="center">
  <img src="https://github.com/Rituraj-commits/BraTs-VNet/blob/main/figs/superpixel_mask.png">
  <br>
  <br>
  <em align="center">Fig 1: Brain Tumour Segmentation </em>
  <br>
</div>

## Getting Started
### Dataset
#### 4D Multimodal MRI dataset 
The dataset contains 750 4D volumes of MRI scans(484 for training and 266 for testing). Since the test set is not publicly available we split the train set into train-val-split. We use 400 scans for training and validation and the rest 84 for evaluation. No data augmentations are applied to the data. The data is stored in NIfTI file format(.nii.gz). A 4D tensor of shape (4,150,240,240) is obtained after reading the data where the 1st dimension denotes the modality(Flair,T1w,t1gd,T2w), 2nd dimension denotes the number of slices and the 3rd and 4th dimesion denotes the width and height respectively. We crop each modality to (32,128,128) for computational purpose and stack each modality along the 0th axis. The segmentation masks contain 3 classes - ED,ET,NET/NCR. We resize and stack each class to form a tensor of shape (3,32,128,128).

### Experimental Details
#### Loss functions
We use Dice loss as the objective function to train the model.
<div align="center">
  <img src="https://github.com/Rituraj-commits/BraTs-VNet/blob/main/figs/formula1.png">
  <br>
  <br>
  <em align="center"></em>
  <br>
</div>

#### Training
We use Adam optimizer for optimizing the objective function. The learning rate is initially set to 0.001 and halved after every 100 epochs. We train the network until 300 epochs and the best weights are saved accordingly. We use NVIDIA Tesla P100 with 16 GB of VRAM to train the model.

### Quantative Results
We evaluate the model on the basis of Dice Score Coefficient(DSC) and Intersection over Union(IoU) over three classes (WT+TC+ET).
<div align="center">
  <img src="https://github.com/Rituraj-commits/BraTs-VNet/blob/main/figs/result1.png">
  <br>
  <br>
  <em align="center"></em>
  <br>
</div>

### Qualitative Results
<div align="center">
  <img src="https://github.com/Rituraj-commits/BraTs-VNet/blob/main/figs/results.png">
  <br>
  <br>
  <em align="center">Fig 1: Brain Complete Tumour Segmentation(blue indicates ground truth segmentation and red indicates predicted segmentation)  </em>
  <br>
</div>

### Statistical Inference
<div align="center">
  <img src="https://github.com/Rituraj-commits/BraTs-VNet/blob/main/figs/Plot%2061.png">
  <br>
  <br>
  <em align="center">Fig 1: Validation Dice Score Coefficient(DSC) </em>
  <br>
</div>


<div align="center">
  <img src="https://github.com/Rituraj-commits/BraTs-VNet/blob/main/figs/Plot%2062.png">
  <br>
  <br>
  <em align="center">Fig 2: Validation Dice Loss </em>
  <br>
</div>

### Dependencies

* SimpleITK 2.0.2
* Pytorch 1.8.0
* CUDA 10.2
* TensorBoard 2.5.0

### Installing

```
 pip install SimpleITK
```
```
 pip install tensorboard
```

### Execution


```
 python train.py
```
```train.py``` contains code for training the model and saving the weights.

```loader.py``` contains code for dataloading and train-test split.

```utils.py``` contains utility functions.

```evaluate.py``` contains code for evaluation.

## Acknowledgments

[1] [BraTS 3D UNet](https://www.kaggle.com/polomarco/brats20-3dunet-3dautoencoder)

[2] [VNet](https://github.com/black0017/MedicalZooPytorch)
