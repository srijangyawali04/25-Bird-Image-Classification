# Image Classification Model

This repository contains the implementation of an image classification model using PyTorch. The model utilizes separable convolutions and squeeze-and-excitation (SE) blocks to enhance performance.

## Table of Contents

- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Saving the Model](#saving-the-model)
- [Plotting Metrics](#plotting-metrics)
- [License](#license)


# Data Preprocessing

This document describes the steps required to prepare your dataset for training and validation.

## 1. Dataset Structure

The Seen dataset was provided in the structure as below:
Seen Datasets/
├── train/
│   ├── Common-Kingfisher/
│   │   ├── Common-Kingfisher_2.jpg
│   │   ├── Common-Kingfisher_3.jpg
│   │   └── ...
│   ├── Cattle-Egret/
│   │   ├── Cattle-Egret_1.jpg
│   │   ├── Cattle-Egret_3.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── Common-Kingfisher/
    │   ├── Common-Kingfisher_1.jpg
    │   ├── Common-Kingfisher_4.jpg
    │   └── ...
    ├── Cattle-Egret/
    │   ├── Common-Kingfisher_1.jpg
    │   ├── Common-Kingfisher_16.jpg
    │   └── ...
    └── ...

## 2. Mean and Standard Deviation Calculation

This section of the code prepares the datasets and calculates the mean and standard deviation of the pixel values in the dataset. These statistics are essential for normalizing the dataset, which helps in improving the performance and stability of machine learning models.

### 1. Load the Dataset
```python
train_set = ImageFolder(train_dir, transform=tt.Compose([
    tt.ToTensor()
]))

val_set = ImageFolder(val_dir, transform=tt.Compose([
    tt.ToTensor()
]))
```
### 2. Concatenate Datasets

The training and validation datasets are concatenated into a single dataset using ConcatDataset. This combined dataset will be used to calculate the mean and standard deviation.
```python
dataset = ConcatDataset([train_set, val_set])
```

### 3. Create a DataLoader

A DataLoader is created from the concatenated dataset to load the data in batches. This is useful for iterating over the dataset efficiently.
```python
dataset_dl = DataLoader(dataset, batch_size, shuffle=True)
```
`batch_size`: Number of images to load in each batch.
`shuffle`: Whether to shuffle the dataset every epoch.