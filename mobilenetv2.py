import torch 
import numpy as np 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torch.nn.functional as F 
from PIL import Image
import os, sys, time, csv
from torch.utils.data import DataLoader, Dataset  
from torch.autograd import Variable 
import hyperparams as hp 



### MobileNetV2 implementation ###
### Jae Kang ###
### jkang2 ### 
### implementation mostly based on KuangLi's PyTorch rendition of MobileNetv2 ### 
### CIFAR10 dataset with 32 x 32 images require stride of 1 and kernel size of 4 ###
 
### play with hyperparameters to make it work on 32 x 32 dataset given ###
### ZCA whitening may be required ### 


## globals ## 

std_stride = hp.mobile_std_stride


pre_feat_dim_shape = 4096

class BottleNeck(nn.Module):
    # t for expand variable, c for number of out channels, n for num stride layers 
    # s for stride 
    ### Bottleneck as described in MobileNetV2 paper ### 
    ### in the order of 
    def __init__(self, in_planes, out_planes, expansion, stride): 
        super(BottleNeck, self).__init__() 
        self.stride = stride

        planes = expansion * in_planes
        ## basic linear bottleneck as building block of MobileNetV2 ##
        self.conv1 = nn.Conv2d(in_planes, planes , kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)


        self.skip = nn.Sequential()
        if self.stride == 1 and in_planes != out_planes:
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        return

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # print("self.skip(x).shape :", self.skip(x).shape)
        # print("out.shape :", out.shape)
        if self.stride == 1:
            return out 
        else: return out + self.skip(x)


class MobileNetV2(nn.Module):
    ### mobilenetv2 class ### 
    def __init__(self): 
        bottleneck = BottleNeck 
        input_channel = hp.input_size 
        last_channel = hp.num_classes
        self.feature_embed_shape = (hp.batch_size, hp.mb_feat_dim)

        self.inverted_setting = [
            # t, c, n, s
            [1, 8, 1, 1],
            [6, 16, 2, 1],
            [6, 32, 3, 1],
            # [3, 64, 3, 1],
            # [6, 96, 3, 1],
            # [6, 160, 3, 1],
            # [6, 320, 1, 1],
        ]
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.linear = nn.Linear(pre_feat_dim_shape, hp.num_classes)
        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(pre_feat_dim_shape, hp.mb_feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.inverted_setting:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(BottleNeck(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        feature_embedding = out.view(out.size(0), -1)
        # print("feature_embedding.shape :", feature_embedding.shape) ## (batch_size, 2048)
        # feature_embedding right before the final linear layer 
        # pass this through a closs layer 
        label_out = self.linear(feature_embedding)
        #closs_out = self.relu_closs(self.linear_closs(feature_embedding))
        #return closs_out, label_out
        self.feature_embedding = feature_embedding
        return label_out 
    

