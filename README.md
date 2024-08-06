# Image Classification Model

This repository contains the implementation of an image classification model using PyTorch. The model utilizes separable convolutions and squeeze-and-excitation (SE) blocks to enhance performance.

## Table of Contents

- [Data Preparation](#data-preparation)
- [Model Building](#model-building)
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
`ConcatDataset`: A utility from torch.utils.data that concatenates multiple datasets.

### 3. Create a DataLoader

A DataLoader is created from the concatenated dataset to load the data in batches. This is useful for iterating over the dataset efficiently.
```python
dataset_dl = DataLoader(dataset, batch_size, shuffle=True)
```
`batch_size`: Number of images to load in each batch.\
`shuffle`: Whether to shuffle the dataset every epoch.

### 4. Calculate Mean and Standard Deviation

The get_mean_and_std function calculates the mean and standard deviation of the pixel values in the dataset. It iterates through the dataset, computes the sum and squared sum of the pixel values for each channel, and then calculates the mean and standard deviation.

```python
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(dataloader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std
```
`dataloader`: DataLoader instance to load the dataset in batches.\
`channels_sum`: Sum of the pixel values for each channel.\
`channels_squared_sum`: Sum of the squared pixel values for each channel.\
`num_batches`: Total number of batches processed.\
`mean`: Calculated mean of the pixel values for each channel.\
`std`: Calculated standard deviation of the pixel values for each channel.

## Model Building

The model architecture is defined using PyTorch, with a base class for common operations and a specific implementation for the image classification task.

#### 1. Base Class for Image Classification

The `ImageClassificationBase` class contains methods for training and validation steps, as well as for calculating accuracy and logging epoch results.

```python
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)          
        return loss, acc
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)         
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f},train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'],result['train_acc'], result['val_loss'], result['val_acc']))
```

### 2. Squeeze-and-Excitation (SE) Block

The SEBlock class implements the Squeeze-and-Excitation mechanism to recalibrate channel-wise feature responses.
```python
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, num_channels, 1, 1)
        return x * y.expand_as(x)
```
#### Initialization (__init__ method):

`in_channels`: The number of input channels.\
`reduction`: The reduction ratio, which controls the bottleneck in the block. The default value is 16.\
`fc1`: A fully connected layer that reduces the number of channels by the reduction ratio.\
`fc2`: A fully connected layer that restores the number of channels to the original number.\

##### Forward Pass (forward method):

`x`: The input tensor with shape (batch_size, num_channels, height, width).\
`F.adaptive_avg_pool2d(x, 1)`: Applies adaptive average pooling to reduce the spatial dimensions to 1x1, resulting in a tensor of shape (batch_size, num_channels, 1, 1).\
`view(batch_size, num_channels)`: Reshapes the tensor to (batch_size, num_channels) for the fully connected layers.\
`F.relu(self.fc1(y))`: Applies the first fully connected layer followed by a ReLU activation function. This reduces the number of channels by the reduction ratio.\
`torch.sigmoid(self.fc2(y))`: Applies the second fully connected layer followed by a sigmoid activation function. This restores the number of channels to the original number and squashes the output to the range [0, 1].\
`view(batch_size, num_channels, 1, 1)`: Reshapes the tensor back to (batch_size, num_channels, 1, 1).\
`x * y.expand_as(x)`: Multiplies the original input tensor x by the recalibrated weights y, which are broadcasted to match the spatial dimensions of x.