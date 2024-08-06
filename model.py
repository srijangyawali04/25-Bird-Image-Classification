import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out,labels)          
        return loss,acc
    
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

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, use_se_block=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if use_se_block:
        layers.append(SEBlock(out_channels))
    return nn.Sequential(*layers)

def SeparableConv(in_channels, out_channels, use_se_block=False):
    layers = [
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if use_se_block:
        layers.append(SEBlock(out_channels))
    return nn.Sequential(*layers)

def linear(in_features, out_features, dropout_rate=0.3):
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(inplace=True)
    )

class ImgClassifier(ImageClassificationBase):
    def __init__(self, output_dim):
        super(ImgClassifier, self).__init__()

        self.features = nn.Sequential(
            conv(3, 64, stride=2, use_se_block=True),
            nn.MaxPool2d(2),

            SeparableConv(64, 128, use_se_block=True),
            nn.MaxPool2d(2),

            SeparableConv(128, 256, use_se_block=True),
            nn.MaxPool2d(2),

            SeparableConv(256, 512, use_se_block=True),
            nn.MaxPool2d(2),

            SeparableConv(512, 512, use_se_block=True),
            nn.MaxPool2d(2),

            conv(512, 512, use_se_block=True),  
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            linear(512, 1024),  
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  
        x = self.classifier(x)
        return x
