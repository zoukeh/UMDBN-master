import torch
import torch.nn.functional as F
from torch import nn
from scipy.io import savemat
from torchvision import models

LOSS_TP = nn.L1Loss()

EPS = 1e-10


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


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
        out = out * self.res_scale + x1
        return out


class MergeTail(nn.Module):
    def __init__(self, n_feats, out_channels):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, int(n_feats / 4))
        self.conv23 = conv1x1(int(n_feats / 2), int(n_feats / 4))
        self.conv_merge = conv3x3(3 * int(n_feats / 4), out_channels)
        self.conv_tail1 = conv3x3(out_channels, out_channels)
        self.conv_tail2 = conv1x1(n_feats // 2, 3)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge(torch.cat((x3, x13, x23), dim=1)))
        x = self.conv_tail1(x)
        # x = self.conv_tail2(x)
        return x


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False


# This function implements the learnable spectral feature extractor (abreviated as LSFE)
# Input:    Hyperspectral or PAN image
# Outputs:  out1 = features at original resolution, out2 = features at original resolution/2, out3 = features at original resolution/4
class LFE(nn.Module):
    def __init__(self, in_channels):
        super(LFE, self).__init__()
        # Define number of input channels
        self.in_channels = in_channels

        # First level convolutions
        self.conv_64_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn_64_1 = nn.BatchNorm2d(64)
        self.conv_64_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_64_2 = nn.BatchNorm2d(64)

        # LeakyReLU
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, x):
        # First level outputs
        out1 = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out1 = self.bn_64_2(self.conv_64_2(out1))
        return out1


# This function implements the multi-head attention
class NoAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self):
        super().__init__()

    def forward(self, v, k, q, mask=None):
        output = v
        return output


class ScaledDotProductAttentionOnly(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)

        # Reshaping K,Q, and Vs...
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn = F.softmax(attn, dim=-1)

        # Attention output
        output = torch.matmul(attn, v)

        # Reshape output to original format
        output = output.view(b, c, h, w)
        return output


class Transformer(nn.Module):
    def __init__(self, E_in_c, A_in_c, temperature):
        super().__init__()
        self.E_in_c = E_in_c  # Number of input channels = Number of output channels
        self.A_in_c = A_in_c
        # Learnable feature extractors (FE-PAN & FE-HSI)
#        self.LFE_E = LFE(in_channels=self.E_in_c)
#        self.LFE_A = LFE(in_channels=self.A_in_c)

        # Attention
        self.DotProductAttention = ScaledDotProductAttentionOnly(temperature=temperature)


    def forward(self, E):
        #E 端元 A 丰度
        b, c, h, w = E.size(0), E.size(1), E.size(2), E.size(3)
        A = torch.ones(b,c,h,w)
        A = A.cuda()
        # Obtaining Values, Keys, and Queries
#        V = self.LFE_A(A)
#        K = self.LFE_E(E)
#        Q = self.LFE_E(E)
#        V = A
#        K 

        # Obtaining T (Transfered HR Features)
#        T = self.DotProductAttention(V, K, Q)
        T = self.DotProductAttention(A, E, E)

        return T

if __name__=="__main__":

    A = torch.randn(1,64,32,32)
    E = torch.randn(1,64,32,32)

    T = Transformer(64,64,16)
    output = T(E,A)
    print(output.size())





# This function implements the multi-head attention
# class ScaledDotProductAttention(nn.Module):
#     ''' Scaled Dot-Product Attention '''
#
#     def __init__(self, temperature):
#         super().__init__()
#         self.temperature = temperature
#
#     def forward(self, v, k, q, mask=None):
#         # Compute attention
#         attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
#
#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e9)
#
#         # Normalization (SoftMax)
#         attn = F.softmax(attn, dim=-1)
#
#         # Attention output
#         output = torch.matmul(attn, v)
#         return output


# class MultiHeadAttention(nn.Module):
#     ''' Multi-Head Attention module for Hyperspectral Pansharpening (Image Fusion) '''
#
#     def __init__(self, n_head, in_pixels, linear_dim, num_features):
#         super().__init__()
#         # Parameters
#         self.n_head = n_head  # No of heads
#         self.in_pixels = in_pixels  # No of pixels in the input image
#         self.linear_dim = linear_dim  # Dim of linear-layer (outputs)
#
#         # Linear layers
#
#         self.w_qs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for queries
#         self.w_ks = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for keys
#         self.w_vs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for values
#         self.fc = nn.Linear(n_head * linear_dim, in_pixels, bias=False)  # Final fully connected layer
#
#         # Scaled dot product attention
#         self.attention = ScaledDotProductAttention(temperature=in_pixels ** 0.5)
#
#         # Batch normalization layer
#         self.OutBN = nn.BatchNorm2d(num_features=num_features)
#
#     def forward(self, v, k, q, mask=None):
#         # Reshaping matrixes to 2D
#         # q = b, c_q, h*w
#         # k = b, c_k, h*w
#         # v = b, c_v, h*w
#         b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)
#         n_head = self.n_head
#         linear_dim = self.linear_dim
#
#         # Reshaping K, Q, and Vs...
#         q = q.view(b, c, h * w)
#         k = k.view(b, c, h * w)
#         v = v.view(b, c, h * w)
#
#         # Save V
#         output = v
#
#         # Pass through the pre-attention projection: b x lq x (n*dv)
#         # Separate different heads: b x lq x n x dv
#         q = self.w_qs(q).view(b, c, n_head, linear_dim)
#         k = self.w_ks(k).view(b, c, n_head, linear_dim)
#         v = self.w_vs(v).view(b, c, n_head, linear_dim)
#
#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
#
#         if mask is not None:
#             mask = mask.unsqueeze(1)  # For head axis broadcasting.
#
#         # Computing ScaledDotProduct attention for each head
#         v_attn = self.attention(v, k, q, mask=mask)
#
#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         v_attn = v_attn.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)
#         v_attn = self.fc(v_attn)
#
#         output = output + v_attn
#         # output  = v_attn
#
#         # Reshape output to original image format
#         output = output.view(b, c, h, w)
#
#         # We can consider batch-normalization here,,,
#         # Will complete it later
#         output = self.OutBN(output)
#         return output


#######################################
# Hyperspectral Transformer (HSIT) ####
#######################################
# Experimenting with soft attention
# class HyperTransformer(nn.Module):
#     def __init__(self, config):
#         super(HyperTransformer, self).__init__()
#         # Settings
#         self.is_DHP_MS = config["is_DHP_MS"]
#         self.in_channels = config[config["train_dataset"]]["spectral_bands"]
#         self.out_channels = config[config["train_dataset"]]["spectral_bands"]
#         self.factor = config[config["train_dataset"]]["factor"]
#         self.config = config
#
#         # Parameter setup
#         self.num_res_blocks = [16, 4, 4, 4, 4]
#         self.n_feats = 256
#         self.res_scale = 1
#
#         # FE-PAN & FE-HSI
#         self.LFE_HSI = LFE(in_channels=self.in_channels)
#         self.LFE_PAN = LFE(in_channels=1)
#
#         # Dimention of each Scaled-Dot-Product-Attention module
#         lv1_dim = config[config["train_dataset"]]["LR_size"] ** 2
#         lv2_dim = (2 * config[config["train_dataset"]]["LR_size"]) ** 2
#         lv3_dim = (4 * config[config["train_dataset"]]["LR_size"]) ** 2
#
#         # Number of Heads in Multi-Head Attention Module
#         n_head = config["N_modules"]
#
#         # Setting up Multi-Head Attention or Single-Head Attention
#         if n_head == 0:
#             # No Attention #
#             # JUst passing the HR features from PAN image (Values) #
#             self.TS_lv3 = NoAttention()
#             self.TS_lv2 = NoAttention()
#             self.TS_lv1 = NoAttention()
#         elif n_head == 1:
#             ### Scaled Dot Product Attention ###
#             self.TS_lv3 = ScaledDotProductAttentionOnly(temperature=lv1_dim)
#             self.TS_lv2 = ScaledDotProductAttentionOnly(temperature=lv2_dim)
#             self.TS_lv1 = ScaledDotProductAttentionOnly(temperature=lv3_dim)
#         else:
#             ### Multi-Head Attention ###
#             lv1_pixels = config[config["train_dataset"]]["LR_size"] ** 2
#             lv2_pixels = (2 * config[config["train_dataset"]]["LR_size"]) ** 2
#             lv3_pixels = (4 * config[config["train_dataset"]]["LR_size"]) ** 2
#             self.TS_lv3 = MultiHeadAttention(n_head=int(n_head),
#                                              in_pixels=int(lv1_pixels),
#                                              linear_dim=int(config[config["train_dataset"]]["LR_size"]),
#                                              num_features=self.n_feats)
#             self.TS_lv2 = MultiHeadAttention(n_head=int(n_head),
#                                              in_pixels=int(lv2_pixels),
#                                              linear_dim=int(config[config["train_dataset"]]["LR_size"]),
#                                              num_features=int(self.n_feats / 2))
#             self.TS_lv1 = MultiHeadAttention(n_head=int(n_head),
#                                              in_pixels=int(lv3_pixels),
#                                              linear_dim=int(config[config["train_dataset"]]["LR_size"]),
#                                              num_features=int(self.n_feats / 4))
#
#         self.SFE = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)
#
#         ###############
#         ### stage11 ###
#         ###############
#         if config[config["train_dataset"]]["feature_sum"]:
#             self.conv11_headSUM = conv3x3(self.n_feats, self.n_feats)
#         else:
#             self.conv11_head = conv3x3(2 * self.n_feats, self.n_feats)
#
#         self.conv12 = conv3x3(self.n_feats, self.n_feats * 2)
#         self.ps12 = nn.PixelShuffle(2)
#         # Residial blocks
#         self.RB11 = nn.ModuleList()
#         for i in range(self.num_res_blocks[1]):
#             self.RB11.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats,
#                                       res_scale=self.res_scale))
#         self.conv11_tail = conv3x3(self.n_feats, self.n_feats)
#
#         ###############
#         ### stage22 ###
#         ###############
#         if config[config["train_dataset"]]["feature_sum"]:
#             self.conv22_headSUM = conv3x3(int(self.n_feats / 2), int(self.n_feats / 2))
#         else:
#             self.conv22_head = conv3x3(2 * int(self.n_feats / 2), int(self.n_feats / 2))
#         self.conv23 = conv3x3(int(self.n_feats / 2), self.n_feats)
#         self.ps23 = nn.PixelShuffle(2)
#         # Residual blocks
#         self.RB22 = nn.ModuleList()
#         for i in range(self.num_res_blocks[2]):
#             self.RB22.append(ResBlock(in_channels=int(self.n_feats / 2), out_channels=int(self.n_feats / 2),
#                                       res_scale=self.res_scale))
#         self.conv22_tail = conv3x3(int(self.n_feats / 2), int(self.n_feats / 2))
#
#         ###############
#         ### stage33 ###
#         ###############
#         if config[config["train_dataset"]]["feature_sum"]:
#             self.conv33_headSUM = conv3x3(int(self.n_feats / 4), int(self.n_feats / 4))
#         else:
#             self.conv33_head = conv3x3(2 * int(self.n_feats / 4), int(self.n_feats / 4))
#         self.RB33 = nn.ModuleList()
#         for i in range(self.num_res_blocks[3]):
#             self.RB33.append(ResBlock(in_channels=int(self.n_feats / 4), out_channels=int(self.n_feats / 4),
#                                       res_scale=self.res_scale))
#         self.conv33_tail = conv3x3(int(self.n_feats / 4), int(self.n_feats / 4))
#
#         ##############
#         ### FINAL ####
#         ##############
#         self.final_conv = nn.Conv2d(in_channels=self.n_feats + int(self.n_feats / 2) + int(self.n_feats / 4),
#                                     out_channels=self.out_channels, kernel_size=3, padding=1)
#         self.RBF = nn.ModuleList()
#         for i in range(self.num_res_blocks[4]):
#             self.RBF.append(
#                 ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
#         self.convF_tail = conv3x3(self.out_channels, self.out_channels)
#
#         ###############
#         # Batch Norm ##
#         ###############
#         self.BN_x11 = nn.BatchNorm2d(self.n_feats)
#         self.BN_x22 = nn.BatchNorm2d(int(self.n_feats / 2))
#         self.BN_x33 = nn.BatchNorm2d(int(self.n_feats / 4))
#
#         ######################
#         # MUlti-Scale-Output #
#         ######################
#         self.up_conv13 = nn.ConvTranspose2d(in_channels=self.n_feats, out_channels=self.in_channels, kernel_size=3,
#                                             stride=4, output_padding=1)
#         self.up_conv23 = nn.ConvTranspose2d(in_channels=int(self.n_feats / 2), out_channels=self.in_channels,
#                                             kernel_size=3, stride=2, padding=1, output_padding=1)
#
#         ###########################
#         # Transfer Periferal Loss #
#         ###########################
#         self.VGG_LFE_HSI = VGG_LFE(in_channels=self.in_channels, requires_grad=False)
#         self.VGG_LFE_PAN = VGG_LFE(in_channels=1, requires_grad=False)
#
#     def forward(self, X_MS, X_PAN):
#         with torch.no_grad():
#             if not self.is_DHP_MS:
#                 X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bicubic')
#
#             else:
#                 X_MS_UP = X_MS
#
#             # Generating PAN, and PAN (UD) images
#             X_PAN = X_PAN.unsqueeze(dim=1)
#             PAN_D = F.interpolate(X_PAN, scale_factor=(1 / self.factor, 1 / self.factor), mode='bilinear')
#             PAN_UD = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode='bilinear')
#
#         # Extracting T and S at multiple-scales
#         # lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
#         V_lv1, V_lv2, V_lv3 = self.LFE_PAN(X_PAN)
#         K_lv1, K_lv2, K_lv3 = self.LFE_PAN(PAN_UD)
#         Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(X_MS_UP)
#
#         T_lv3 = self.TS_lv3(V_lv3, K_lv3, Q_lv3)
#         T_lv2 = self.TS_lv2(V_lv2, K_lv2, Q_lv2)
#         T_lv1 = self.TS_lv1(V_lv1, K_lv1, Q_lv1)
#
#         # Save feature maps for illustration purpose
#         # feature_dic={}
#         # feature_dic.update({"V": V_lv3.detach().cpu().numpy()})
#         # feature_dic.update({"K": K_lv3.detach().cpu().numpy()})
#         # feature_dic.update({"Q": Q_lv3.detach().cpu().numpy()})
#         # feature_dic.update({"T": T_lv3.detach().cpu().numpy()})
#         # savemat("/home/lidan/Dropbox/Hyperspectral/HyperTransformer/feature_visualization_pavia/soft_attention/multi_head_no_skip_lv3.mat", feature_dic)
#         # exit()
#
#         # Shallow Feature Extraction (SFE)
#         x = self.SFE(X_MS)
#
#         #####################################
#         #### stage1: (L/4, W/4) scale ######
#         #####################################
#         x11 = x
#         # HyperTransformer at (L/4, W/4) scale
#         x11_res = x11
#         if self.config[self.config["train_dataset"]]["feature_sum"]:
#             x11_res = x11_res + T_lv3
#             x11_res = self.conv11_headSUM(x11_res)  # F.relu(self.conv11_head(x11_res))
#         else:
#             x11_res = torch.cat((self.BN_x11(x11_res), T_lv3), dim=1)
#             x11_res = self.conv11_head(x11_res)  # F.relu(self.conv11_head(x11_res))
#         x11 = x11 + x11_res
#         # Residial learning
#         x11_res = x11
#         for i in range(self.num_res_blocks[1]):
#             x11_res = self.RB11[i](x11_res)
#         x11_res = self.conv11_tail(x11_res)
#         x11 = x11 + x11_res
#
#         #####################################
#         #### stage2: (L/2, W/2) scale ######
#         #####################################
#         x22 = self.conv12(x11)
#         x22 = F.relu(self.ps12(x22))
#         # HyperTransformer at (L/2, W/2) scale
#         x22_res = x22
#         if self.config[self.config["train_dataset"]]["feature_sum"]:
#             x22_res = x22_res + T_lv2
#             x22_res = self.conv22_headSUM(x22_res)  # F.relu(self.conv22_head(x22_res))
#         else:
#             x22_res = torch.cat((self.BN_x22(x22_res), T_lv2), dim=1)
#             x22_res = self.conv22_head(x22_res)  # F.relu(self.conv22_head(x22_res))
#         x22 = x22 + x22_res
#         # Residial learning
#         x22_res = x22
#         for i in range(self.num_res_blocks[2]):
#             x22_res = self.RB22[i](x22_res)
#         x22_res = self.conv22_tail(x22_res)
#         x22 = x22 + x22_res
#
#         #####################################
#         ###### stage3: (L, W) scale ########
#         #####################################
#         x33 = self.conv23(x22)
#         x33 = F.relu(self.ps23(x33))
#         # HyperTransformer at (L, W) scale
#         x33_res = x33
#         if self.config[self.config["train_dataset"]]["feature_sum"]:
#             x33_res = x33_res + T_lv1
#             x33_res = self.conv33_headSUM(x33_res)  # F.relu(self.conv33_head(x33_res))
#         else:
#             x33_res = torch.cat((self.BN_x33(x33_res), T_lv1), dim=1)
#             x33_res = self.conv33_head(x33_res)  # F.relu(self.conv33_head(x33_res))
#         x33 = x33 + x33_res
#         # Residual learning
#         x33_res = x33
#         for i in range(self.num_res_blocks[3]):
#             x33_res = self.RB33[i](x33_res)
#         x33_res = self.conv33_tail(x33_res)
#         x33 = x33 + x33_res
#
#         #####################################
#         ############ Feature Pyramid ########
#         #####################################
#         x11_up = F.interpolate(x11, scale_factor=4, mode='bicubic')
#         x22_up = F.interpolate(x22, scale_factor=2, mode='bicubic')
#         xF = torch.cat((x11_up, x22_up, x33), dim=1)
#
#         #####################################
#         ####  Final convolution   ###########
#         #####################################
#         xF = self.final_conv(xF)
#         xF_res = xF
#
#         # Final resblocks
#         for i in range(self.num_res_blocks[4]):
#             xF_res = self.RBF[i](xF_res)
#         xF_res = self.convF_tail(xF_res)
#         x = xF + xF_res
#
#         #####################################
#         #      Transfer Periferal Loss      #
#         #####################################
#         # v_vgg_lv1, v_vgg_lv2, v_vgg_lv3 = self.VGG_LFE_HSI(X_MS_UP)
#         # q_vgg_lv1, q_vgg_lv2, q_vgg_lv3 = self.VGG_LFE_PAN(X_PAN)
#         # loss_tp = LOSS_TP(V_lv1, v_vgg_lv1) + LOSS_TP(V_lv2, v_vgg_lv2) + LOSS_TP(V_lv3, v_vgg_lv3)
#         # loss_tp = loss_tp + LOSS_TP(Q_lv1, q_vgg_lv1) + LOSS_TP(Q_lv2, q_vgg_lv2) + LOSS_TP(Q_lv3, q_vgg_lv3)
#
#         Phi_lv1, Phi_lv2, Phi_lv3 = self.LFE_HSI(x.detach())
#         Phi_T_lv3 = self.TS_lv3(V_lv3, K_lv3, Phi_lv3)
#         Phi_T_lv2 = self.TS_lv2(V_lv2, K_lv2, Phi_lv2)
#         Phi_T_lv1 = self.TS_lv1(V_lv1, K_lv1, Phi_lv1)
#         loss_tp = LOSS_TP(Phi_T_lv1, T_lv1) + LOSS_TP(Phi_T_lv2, T_lv2) + LOSS_TP(Phi_T_lv3, T_lv3)
#
#         #####################################
#         #       Output                      #
#         #####################################
#         x13 = self.up_conv13(x11)
#         x23 = self.up_conv23(x22)
#         output = {"pred": x,
#                   "x13": x13,
#                   "x23": x23,
#                   "tp_loss": loss_tp}
#         return output



