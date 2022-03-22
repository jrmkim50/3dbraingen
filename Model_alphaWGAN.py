import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F

#***********************************************
#Encoder and Discriminator has same architecture
#***********************************************
class Discriminator(nn.Module):
    def __init__(self, channel=512, resolution=64):
        super(Discriminator, self).__init__()        
        self.channel = channel
        self._resolution = resolution
        
        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(channel//8)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)
        self.conv5 = nn.Conv3d(channel, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

        _inchannels = 2*2*5 if self._resolution == 64 else 2*2*5*512
        self.dense1 = nn.Linear(_inchannels, channel)
        self.dense2 = nn.Linear(channel, 1)

    def forward(self, x):
        # input: 64,64,160 - conv1 -> 32,32,80 - conv2 -> 16,16,40 - conv3 -> 8,8,20
        #     - conv4 -> 4,4,10 - conv5 -> 2,2,5 - dense1 -> 512 - dense2 -> 1
        # input: 32,32,80 - conv1 -> 16,16,40 - conv2 -> 8,8,20 - conv3 -> 4,4,10
        #     - conv4 -> 2,2,5 - dense1 -> 512 - dense2 -> 1
        h1 = self.leaky_relu(self.bn1(self.conv1(x)))
        h2 = self.leaky_relu(self.bn2(self.conv2(h1)))
        h3 = self.leaky_relu(self.bn3(self.conv3(h2)))
        if self._resolution == 64:
            h4 = self.leaky_relu(self.bn4(self.conv4(h3)))
            h5 = self.conv5(h4)
            h5 = h5.view(-1)
            h5 = self.leaky_relu(self.dense1(h5))
            h5 = self.dense2(h5)
            return h5
        else:
            h4 = self.conv4(h3)
            h4 = h4.view(-1)
            h4 = self.leaky_relu(self.dense1(h4))
            h4 = self.dense2(h4)
            return h4
    
class Code_Discriminator(nn.Module):
    def __init__(self, code_size=100,num_units=750):
        super(Code_Discriminator, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(code_size, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l2 = nn.Sequential(nn.Linear(num_units, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = nn.Linear(num_units, 1)
        
    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        output = h3
        return output

class Generator(nn.Module):
    def __init__(self, noise:int=1000, channel:int=64, resolution=64):
        super(Generator, self).__init__()
        _c = channel
        self._resolution = resolution
        
        self.noise = noise
        self.fc = nn.Linear(self.noise,512*2*2*5)
        self.bn1 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv2 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv3 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(_c*2)
        
        self.tp_conv4 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(_c)
        
        self.tp_conv5 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1)
        if self._resolution == 64:
            self.tp_conv5 = nn.Sequential(
                nn.Conv3d(_c, _c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(_c),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1)
            )
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=2)
        
    def forward(self, noise):
        # 1000 - fc -> 512,2,2,5 - conv2 -> 256,4,4,10 - conv3 -> 128,8,8,20 
        #     - conv4 -> 64,16,16,40 - conv5 -> 64,32,32,80 - conv6 -> 1,64,64,160
        # 1000 - fc -> 512,2,2,5 - conv2 -> 256,4,4,10 - conv3 -> 128,8,8,20 
        #     - conv4 -> 64,16,16,40 - conv5 -> 1,32,32,80
        noise = noise.view(-1, self.noise)
        h = self.fc(noise)
        h = h.view(-1,512,2,2,5)
        h = self.relu(self.bn1(h))

        h = self.upsample(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))
        
        h = self.upsample(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))

        h = self.upsample(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = self.upsample(h,scale_factor = 2)
        h = self.tp_conv5(h)

        h = self.tanh(h)

        return h