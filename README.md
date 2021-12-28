# BraTS(Brain Tumour Segmentation) using V-Net

This project is an approach to detect brain tumours using BraTS 2016,2017 dataset.

## Description

BraTS is a dataset which provides multimodal 3D brain MRIs annotated by experts. Each Magnetic Resonance Imaging(MRI) scan consists of 4 different modalities(Flair,T1w,t1gd,T2w).
Expert annotations are provided in the form of segmentation masks to detect 3 classes of tumour - edema(ED),enhancing tumour(ET),necrotic and non-enhancing tumour(NET/NCR). The dataset is challening in terms of the complex and heterogeneously-located targets.
We use Volumetric Network(V-Net) which is a 3D Fully Convolutional Network(FCN) for segmentation of 3D medical images. We use Dice Loss as the objective function for the present scenario. Future implementation will include Hausdorff Loss for better boundary segmentations.

## Getting Started

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

## Acknowledgments

* [BraTS 3D UNet](https://www.kaggle.com/polomarco/brats20-3dunet-3dautoencoder)
* [VNet](https://github.com/black0017/MedicalZooPytorch)
