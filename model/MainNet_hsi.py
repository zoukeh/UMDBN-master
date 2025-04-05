import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class CSA_Block(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1):
        super(CSA_Block, self).__init__()
        self.down_ratio=8
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv3x3(in_channels, self.out_channels,stride)
        self.conv2 = conv3x3(self.out_channels, self.out_channels,stride)
        #ca
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv3 = conv3x3(self.out_channels, int(self.out_channels/self.down_ratio),stride)
        self.conv4 = conv3x3(int(self.out_channels/self.down_ratio), self.out_channels, stride)
        #sa
        self.conv5 =conv3x3(self.out_channels,1,stride)
        self.conv6 =conv3x3(self.in_channels,self.out_channels)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        u = self.conv2(F.relu(self.conv1(x)))
        M_CA = self.sigmod(self.conv4(F.relu(self.conv3(self.gap(u)))))
        M_SA = self.sigmod(self.conv5(u))
        U_CA = u*M_CA
        U_SA = u*M_SA
        # if self.in_channels != self.out_channels:
            # x = self.conv6(x)
        out = U_SA+U_CA+x
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out * self.res_scale + x1)
        return out

class SFE(nn.Module):
    def __init__(self, num_res_blocks, input_bands,n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(input_bands, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            
        self.conv_tail = conv3x3(n_feats, n_feats)
        # self.conv_tail = CSA_Block(n_feats,n_feats)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return self.relu(x)

class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, 2)
        self.conv_merge1 = conv3x3(n_feats*2, n_feats)
        self.conv_merge2 = conv3x3(n_feats*2, n_feats)

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12), dim=1) ))

        return x1, x2

class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        self.conv13 = conv1x1(n_feats, n_feats)

        self.conv23 = conv1x1(n_feats, n_feats)
        self.conv32 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge3 = conv3x3(n_feats*3, n_feats)

        self.conv_merge_tail = conv3x3(n_feats, n_feats)
    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x3 = F.relu(self.conv_merge_tail(self.conv_merge3(torch.cat((x3, x13, x23), dim=1) )))
        
        return x3

class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        # self.conv13 = conv1x1(n_feats, n_feats)
        # self.conv23 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats*3, n_feats)
        self.conv13 = CSA_Block(n_feats,n_feats)
        self.conv23 = CSA_Block(n_feats,n_feats)
        # self.conv_merge = CSA_Block(n_feats*3,n_feats)

        # self.conv_tail1 = conv3x3(n_feats, n_feats//2)
        # self.conv_tail2 = conv1x1(n_feats//2, 128)
        # self.conv_tail1 = conv3x3(n_feats, n_feats)
        self.conv_tail1 = CSA_Block(n_feats,n_feats)
        self.conv_tail2 = conv1x1(n_feats, 128)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge(torch.cat((x3, x13, x23), dim=1) ))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        x = torch.clamp(x, -1, 1)
        
        return x

class MainNet(nn.Module):
    def __init__(self, num_res_blocks, input_bands, n_feats, hsi_channels,res_scale):
        super(MainNet, self).__init__()
        self.num_res_blocks = num_res_blocks ### a list containing number of resblocks of different stages
        self.n_feats = n_feats
        self.input_bands = input_bands #x的通道数
        self.SFE = SFE(self.num_res_blocks[0], input_bands, self.n_feats, res_scale)

        ### stage11
        # self.conv11_head = conv3x3(self.n_feats*2,self.n_feats)
        # self.conv11_head = conv3x3(out_channels+self.n_feats, self.n_feats)
        self.conv11_head = conv3x3(2*self.n_feats, self.n_feats)
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats,
                res_scale=res_scale))
        self.relu = nn.ReLU(inplace=True)
        self.conv11_tail = conv3x3(n_feats, hsi_channels)
        # self.conv11_tail = CSA_Block(self.n_feats,self.n_feats)
        ### subpixel 1 -> 2
        # self.conv12 = conv3x3(n_feats, n_feats*4)
        # self.ps12 = nn.PixelShuffle(2)

        ### stage21, 22
        # self.conv21_head = conv3x3(n_feats, n_feats)
        # self.conv21_head = CSA_Block(n_feats, n_feats)
        # self.conv22_head = conv3x3(128+n_feats, n_feats)
        #
        # self.ex12 = CSFI2(n_feats)
        #
        # self.RB21 = nn.ModuleList()
        # self.RB22 = nn.ModuleList()
        # for i in range(self.num_res_blocks[2]):
        #     self.RB21.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
        #         res_scale=res_scale))
        #     self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
        #         res_scale=res_scale))
        #
        # self.conv21_tail = conv3x3(n_feats, n_feats)
        # self.conv22_tail = conv3x3(n_feats, n_feats)
        #
        # ### subpixel 2 -> 3
        # self.conv23 = conv3x3(n_feats, n_feats*4)
        # self.ps23 = nn.PixelShuffle(2)
        #
        # ### stage31, 32, 33
        # # self.conv31_head = conv3x3(n_feats, n_feats)
        # # self.conv32_head = conv3x3(n_feats, n_feats)
        # self.conv31_head = CSA_Block(n_feats, n_feats)
        # self.conv32_head = CSA_Block(n_feats, n_feats)
        # self.conv33_head = conv3x3(128+n_feats, n_feats)
        #
        # self.ex123 = CSFI3(n_feats)
        #
        # self.RB33 = nn.ModuleList()
        # for i in range(self.num_res_blocks[3]):
        #     self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
        #         res_scale=res_scale))
        #
        # # self.conv31_tail = conv3x3(n_feats, n_feats)
        # # self.conv32_tail = conv3x3(n_feats, n_feats)
        # # self.conv33_tail = conv3x3(n_feats, n_feats)
        # self.conv31_tail = CSA_Block(n_feats, n_feats)
        # self.conv32_tail = CSA_Block(n_feats, n_feats)
        # self.conv33_tail = CSA_Block(n_feats, n_feats)
        # self.merge_tail = MergeTail(n_feats)

    def forward(self, x, S=None, T_lv3=None):
        ### shallow feature extraction

        x = self.SFE(x)

        # lr2 = F.interpolate(x, scale_factor=2, mode='bicubic')
        # lr4 = F.interpolate(x, scale_factor=4, mode='bicubic')
        ### stage11
        # x = self.SFE(x)
        x11 = x  #原始

        ### soft-attention
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.relu(self.conv11_head(x11_res))#F.relu(self.conv11_head(x11_res))
        x11_res = x11_res * S
        x11 = x11 + x11_res

        x11_res = x11
        # x11 = self.SFE(x11)
        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        # x11_res = self.conv11_tail(x11_res)
        # x11 = x11 + x11_res
        x11 = x11 +x11_res
        x11 = self.conv11_tail(x11)

        return x11

if __name__=="__main__":
    mat = scipy.io.loadmat("Z:/chikusei/test/chikusei_100.mat")
    print(mat['ref'])
