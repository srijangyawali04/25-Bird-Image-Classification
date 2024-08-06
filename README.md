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

train_set = ImageFolder(train_dir, transform=tt.Compose([
    tt.ToTensor()
]))

val_set = ImageFolder(val_dir, transform=tt.Compose([
    tt.ToTensor()
]))

