import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import utils

class LTE3DConvLayer(nn.Module):
    def __init__(self,in_size,out_size,kernal=(1,1,1),padding= (0,0,0)):
        super(LTE3DConvLayer, self).__init__()
        self.conv3d = nn.Sequential(nn.Conv3d(in_channels=in_size,out_channels=out_size,kernel_size=kernal,stride=(1,1,1),padding=padding),
                                    # nn.BatchNorm3d(out_size),
                                    nn.LeakyReLU(0.2))
    def forward(self,inputs):
        outputs = self.conv3d(inputs)
        # outputs =  tensor_reshape(outputs)
        return outputs

def tensor_reshape(outputs):
    outputs=outputs.view(outputs.size(0),outputs.size(1)*outputs.size(2),outputs.size(3),outputs.size(3))
    return outputs


# Input:    Hyperspectral or PAN image
# Outputs:  out1 = features at original resolution, out2 = features at original resolution/2, out3 = features at original resolution/4
class LTE_hsi(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(LTE_hsi, self).__init__()
        # Define number of input channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.size = 160
        # First level convolutions
        self.conv_64_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=128, kernel_size=7, padding=3)
        self.bn_64_1 = nn.BatchNorm2d(128)
        self.conv_64_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_64_2 = nn.BatchNorm2d(128)
        self.conv_64_128 = nn.Conv2d(in_channels=128,out_channels=self.out_channels,kernel_size=1)
        self.bn_64_3 = nn.BatchNorm2d(self.out_channels)
        # # Second level convolutions
        # self.conv_128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.bn_128_1 = nn.BatchNorm2d(128)
        # self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.bn_128_2 = nn.BatchNorm2d(128)
        #
        #
        # # Third level convolutions
        # self.conv_256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.bn_256_1 = nn.BatchNorm2d(256)
        # self.conv_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.bn_256_2 = nn.BatchNorm2d(256)
        # self.conv_256_128 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        # self.bn_256_3 = nn.BatchNorm2d(128)
        # Max pooling
        # self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.MaxPool2x21 = nn.AdaptiveMaxPool2d((int(self.size),int(self.size)))
        # self.MaxPool2x22 = nn.AdaptiveMaxPool2d((int(self.size/2),int(self.size/2)))
        # self.MaxPool2x23 = nn.AdaptiveMaxPool2d((int(self.size/4),int(self.size/4)))
        # LeakyReLU
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, x):
        # First level outputs
        # x_res = x
        out1 = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out1 = self.LeakyReLU(self.bn_64_2(self.conv_64_2(out1)))
        out_lv1 = self.LeakyReLU(self.bn_64_3(self.conv_64_128(self.conv_64_2(out1))))
        # out1 = self.LeakyReLU(self.conv_64_1(x))
        # out1 = self.LeakyReLU(self.conv_64_2(out1))
        # out_lv1 = self.LeakyReLU(self.conv_64_128(self.conv_64_2(out1)))
        # # Second level outputs
        # out1_mp = self.MaxPool2x2(self.LeakyReLU(out1))
        # out2 = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out1_mp)))
        # out2 = self.bn_128_2(self.conv_128_2(out2))
        #
        # # Third level outputs
        # out2_mp = self.MaxPool2x2(self.LeakyReLU(out2))
        # out3 = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out2_mp)))
        # out3 = self.bn_256_2(self.conv_256_2(out3))
        # out_lv3 = self.bn_256_3(self.conv_256_128(self.conv_256_2(out3)))
        return out_lv1

class Conv3DBlock(nn.Module):
    def __init__(self,in_channels,mid_channels,kernal = (1,1,1),padding = (0,0,0)):
        super(Conv3DBlock,self).__init__()
        self.conv3d1 = LTE3DConvLayer(in_channels,mid_channels,kernal,padding)
        # self.bn3d = nn.BatchNorm3d(1)
        # self.conv3d2 = LTE3DConvLayer(mid_channels,mid_channels)
        # self.conv3d3 = LTE3DConvLayer(mid_channels,out_channels)
    def forward(self,inputs):
        inputs = inputs.unsqueeze(1)
        outputs =self.conv3d1(inputs)
        # outputs = self.bn3d(outputs)
        # outputs = self.conv3d2(outputs)
        # outputs = self.conv3d3(outputs)
        outputs = tensor_reshape(outputs)
        return outputs

class LTE(nn.Module):
    def __init__(self, in_channels,out_channels,ratio):
        super(LTE, self).__init__()
        # Define number of input channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.conv3d1 = Conv3DBlock(1,1,2)
        # self.conv3d2 =Conv3DBlock(1,1,2)
        # self.conv3d3 = Conv3DBlock(1,1,2)
        # First level convolutions
        self.conv_64_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=128, kernel_size=7, padding=3)
        self.bn_64_1 = nn.BatchNorm2d(128)
        self.conv_64_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_64_2 = nn.BatchNorm2d(128)

        # Second level convolutions
        self.conv_128_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_1 = nn.BatchNorm2d(128)
        self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_2 = nn.BatchNorm2d(128)

        # Third level convolutions
        self.conv_256_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_256_1 = nn.BatchNorm2d(128)
        self.conv_256_2 = nn.Conv2d(in_channels=128, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.bn_256_2 = nn.BatchNorm2d(self.out_channels)

        #fourth
        self.conv_256_3 = nn.Conv2d(in_channels=self.out_channels, out_channels=128, kernel_size=3, padding=1)
        self.bn_256_3 = nn.BatchNorm2d(128)
        self.conv_256_4 = nn.Conv2d(in_channels=128, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.bn_256_4 = nn.BatchNorm2d(self.out_channels)

        # Max pooling
        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # LeakyReLU
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.0)
        self.ratio = ratio
    def forward(self, x):
        # First level outputs
        # x = x.unsqueeze(1)
        # out1 = self.conv3d1(x)
        # out1 = self.conv3d2(out1)
        # out1 = self.conv3d3(out1)
        # out1 = tensor_reshape(out1)
        out1 = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out1 = self.bn_64_2(self.conv_64_2(out1))
        # out1 = self.LeakyReLU(self.conv_64_1(x))
        # out1 = self.conv_64_2(out1)
        # Second level outputs
        # out1 = out1.unsqueeze(1)
        # out1_2 = self.conv3d2(out1)
        out1_2 = out1
        out1_mp = self.MaxPool2x2(self.LeakyReLU(out1_2))
        out2 = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out1_mp)))
        out2 = self.bn_128_2(self.conv_128_2(out2))
        # out2 = self.LeakyReLU(self.conv_128_1(out1_mp))
        # out2 = self.conv_128_2(out2)
        # Third level outputs
        # out2 = out2.unsqueeze(1)
        # out2_3 = self.conv3d3(out2)
        out2_3 = out2
        out2_mp = self.MaxPool2x2(self.LeakyReLU(out2_3))
        out3 = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out2_mp)))
        out3 = self.bn_256_2(self.conv_256_2(out3))
        # out3 = self.LeakyReLU(self.conv_256_1(out2_mp))
        # out3 = self.conv_256_2(out3)
        if(self.ratio==8):
            # out3 = self.conv3d3(out3)
            out3 = self.MaxPool2x2(self.LeakyReLU(out3))
            out3 = self.LeakyReLU(self.bn_256_3(self.conv_256_3(out3)))
            out3 = self.bn_256_4(self.conv_256_4(out3))
            # out3 = self.LeakyReLU(self.conv_256_3(out3))
            # out3 = self.conv_256_4(out3)

        return out3