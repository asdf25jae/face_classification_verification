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


## resnet model ## 
## resnet.py ## 
### implementation mostly based on KuangLi's PyTorch rendition of ResNet ### 

batch_size = hp.batch_size
kernel_size = hp.kernel_size
res_block_ksize = hp.res_block_ksize
relu = F.relu
blocks_per_layer = hp.blocks_per_layer
# stride = hp.stride
avg_pool_stride = hp.avg_pool_stride
padding = hp.padding
num_sizes = hp.num_sizes # default "hidden" layer sizes for ResNet 
final_layer_kernel_size = hp.final_layer_kernel_size
feat_dim = hp.feat_dim
num_classes = hp.num_classes
rgb_channels = hp.rgb_channels
blocks_per_layer_50 = hp.blocks_per_layer_50 # blocks per layer number for ResNet50

## weight init function ## 

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


## Basic bottleneck class for ResNet50 ## 

class Bottleneck(nn.Module):
    #expansion = 4
    def __init__(self, in_channels, channels, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, self.expansion*channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels))

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out = self.bn3(self.conv3(out2))
        out += self.skip(x)
        out = F.relu(out)
        return out


## basic block, for resnet ## 

class ResBlock(nn.Module):
    # expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        # in_channels = in_planes
        # out_channels = planes : how many channels produced by convolution
        super(ResBlock, self).__init__()
        self.expansion = 1
        self.conv1_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1_layer = nn.BatchNorm2d(out_channels)
        self.conv2_layer = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2_layer = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == self.expansion*out_channels:
            self.skip = nn.Sequential()
        else: 
            self.skip = nn.Sequential(nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=res_block_ksize, 
                    stride=stride, bias=False), nn.BatchNorm2d(self.expansion*out_channels))


    def forward(self, x): 
        first_out = F.relu(self.bn1_layer(self.conv1_layer(x)))
        second_out = self.bn2_layer(self.conv2_layer(first_out))
        res_skip = self.skip(x)
        # print("res_skip.shape :", res_skip.shape)
        # print("second_out.shape :", second_out.shape)
        return F.relu(second_out + res_skip)


class ResNet(nn.Module):
    # ResNet architecture based on paper 


    def __init__(self, res_block, num_blocks, num_classes, num_sizes): 
        ## res_block = Residual Block : ResBlock
        ## num_blocks = total number of residual blocks in this ResNet implementation : int*
        ## num_classes = output size, total number of classes we're classifying to : int 
        ## hidden_sizes = array of dimensions of the hidden layers : int*
        super(ResNet, self).__init__() 
        self.in_channels = 64
        self.feature_embed_shape = (batch_size, num_sizes[-1]) ## for ResNet50 specifically 
        self.hiddens =  num_sizes + [num_classes]   
        self.num_blocks = num_blocks
        self.hidden_sizes = num_sizes
        self.layers = []
        self.expansion = 4
        ## define conv1, bn and proceeding layers 
        self.conv1_layer = nn.Conv2d(rgb_channels, self.in_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1_layer = nn.BatchNorm2d(self.in_channels)
        self.layer1 = self.create_layer(res_block, 64, num_blocks[0], stride=1)
        self.layer2 = self.create_layer(res_block, 128, num_blocks[1], stride=1)
        self.layer3 = self.create_layer(res_block, 256, num_blocks[2], stride=1)
        self.layer4 = self.create_layer(res_block, 512, num_blocks[3], stride=1)
        self.res_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        # self.res_layers = [self.layer1, self.layer2, self.layer3]
        self.linear = nn.Linear(num_sizes[-1]*self.expansion, num_classes)
        self.layers_temp = [self.conv1_layer, self.bn1_layer] + self.res_layers + [self.linear]
        self.layers = nn.Sequential(*self.layers_temp) 
        self.linear_label = nn.Linear(self.feature_embed_shape[1], num_classes, bias=False)
        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.feature_embed_shape[1], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)

    def create_layer(self, block, num_channels, num_blocks, stride): 
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        # now fill in with appropriate blocks 
        for stride in strides: 
            n_block = block(self.in_channels, num_channels, stride)
            self.in_channels = num_channels * self.expansion
            layers.append(n_block)
        return nn.Sequential(*layers)

    def forward(self, x): 
        self.x = x 
        out = F.relu(self.bn1_layer(self.conv1_layer(x)))
        for layer in self.res_layers:
            out = layer(out)
        out = F.avg_pool2d(out, [out.size(2), out.size(3)], stride=avg_pool_stride) #pooling 
        # reshape our feature embedding 
        feat_embedding = out.view(out.shape[0], -1)
        closs_output = self.relu_closs(self.linear_closs(feat_embedding))
        # print("feat_embedding.shape :", feat_embedding.shape) # (batch_size = 64), 512 for ResNet50 
        label_output = self.linear_label(feat_embedding) / torch.norm(self.linear_label.weight, dim=1)
        return closs_output, label_output








## Xavier weight initialization for resnet, and convolutional layers ## 
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight.data)

def ResNet34():
    # helper method to return a ResNet34 with given parameters 
    out = ResNet(ResBlock, blocks_per_layer, num_classes, num_sizes)
    out.apply(init_weights) # initialize the 
    return out

def ResNet50():
    model = ResNet(Bottleneck, blocks_per_layer_50, num_classes, num_sizes)
    model.apply(init_weights)
    return model 


def main(): 
    #net = ResNet34() # our resnet model 
    return None 

if __name__ == "__main__": 
    main()