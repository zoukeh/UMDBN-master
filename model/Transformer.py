import torch
import torch.nn.functional as F
from torch import nn
from scipy.io import savemat
from torchvision import models

LOSS_TP = nn.L1Loss()

EPS = 1e-10

# def upper_triangular_to_vector_4d(tensor_4d):
#     # 获取tensor的形状
#     M, N, P, Q = tensor_4d.size()
#
#     # 生成上三角索引的掩码
#     mask = torch.triu(torch.ones(M, N), diagonal=1).bool()
#
#     # 对每个二维矩阵执行上述处理
#     vectorized_results = []
#     for i in range(P):
#         for j in range(Q):
#             matrix = tensor_4d[:, :, i, j]
#             upper_triangular_elements = matrix[mask]
#             vectorized_result = torch.reshape(upper_triangular_elements, (-1, 1))
#             vectorized_results.append(vectorized_result)
#
#     # 将结果拼接成一个tensor
#     final_result = torch.cat(vectorized_results, dim=1)
#
#     # 将处理后的tensor重新组装成四维tensor，保持后两个维度不变
#     final_result = final_result.view((M-1)*N//2, P, Q)
#
#     return final_result
# def upper_triangular_to_vector_higher_dim(matrix):
#     """
#     将高维数组的最后两维的上三角部分展平成向量
#
#     参数：
#     - matrix: 输入的高维数组
#
#     返回：
#     - flattened_vector: 展平后的向量
#     """
#     if len(matrix.shape) < 2:
#         raise ValueError("输入数组的维度必须至少为2")
#
#     last_dim_size = matrix.shape[-1]
#     second_last_dim_size = matrix.shape[-2]
#
#     flattened_size = second_last_dim_size * (second_last_dim_size - 1) // 2
#
#     flattened_vector = torch.zeros(matrix.shape[:-2] + (flattened_size, 1),
#                                    dtype=matrix.dtype, device=matrix.device)
#
#     index = 0
#     for i in range(second_last_dim_size):
#         for j in range(i + 1, last_dim_size):
#             flattened_vector[..., index, 0] = matrix[..., i, j]
#             index += 1
#
#     return flattened_vector

def upper_half_first2(tensor_4d):
    # 获取tensor的形状
    M, N, P,Q = tensor_4d.size()

    # 生成上半部分索引的掩码
    mask = torch.triu(torch.ones(M, N), diagonal=0).bool()

    # 对前两个维度进行操作
    tensor_2d = tensor_4d.view(M, N, P*Q )

    # 使用掩码获取上半部分的元素
    upper_half_elements = tensor_2d[mask, :]

    # 将上半部分的元素拼接成向量
    vectorized_result = upper_half_elements.view(-1, P,Q)

    return vectorized_result


def upper_half_last2(tensor_4d):
    # 获取tensor的形状
    M, P, Q = tensor_4d.size()

    # 生成上半部分索引的掩码
    mask = torch.triu(torch.ones(P, Q), diagonal=0).bool()

    # 对后两个维度进行操作
    tensor_2d = tensor_4d.view(M, -1, Q)

    # 使用掩码获取上半部分的元素
    upper_half_elements = tensor_2d[:, mask]

    # 将上半部分的元素拼接成向量
    vectorized_result = upper_half_elements.view(M, -1)

    return vectorized_result

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
        # self.endmenber = endmenber
    def forward(self,  k, q, endmember):
        b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)

        # Reshaping K,Q, and Vs...

        # v = v.view(b, c, h*w)

        # Compute attention
        # q = torch.mul(q,v)
        # k = torch.mul(k,v)
        if endmember:
            q = q.view(b, c, h * w)
            k = k.view(b, c, h * w)
            k = k.transpose(-2,-1)
            attn = torch.matmul(q / self.temperature, k)
            attn = upper_half_last2(attn)
            output = F.softmax(attn, dim=-1)
            # output = attn
        else:
            k = k.transpose(0,1)
            attn = torch.matmul(q / self.temperature, k)
            attn = upper_half_first2(attn)
            output = F.softmax(attn, dim=0)
            # output=attn
            #对前两个维度进行处理
        # attn = torch.matmul(q / self.temperature, k)
        # # print(attn.size())
        # # if mask is not None:
        # #     attn = attn.masked_fill(mask == 0, -1e9)
        # # attn = attn*v
        # # Normalization (SoftMax)
        # attn = F.softmax(attn, dim=-1)
        #
        # # Attention output
        # # output = torch.mul(attn, v)
        # # output = attn.view(b, c, h, w)
        # output = upper_triangular_to_vector_higher_dim(attn)
        # # Reshape output to original format
        # # output = attn.view(b, c, h, w)
        return output

#class Transformer(nn.Module):
#    def __init__(self, E_in_c, A_in_c, temperature):
#        super().__init__()
#        self.E_in_c = E_in_c  # Number of input channels = Number of output channels
#        self.A_in_c = A_in_c
#        # Learnable feature extractors (FE-PAN & FE-HSI)
##        self.LFE_E = LFE(in_channels=self.E_in_c)
##        self.LFE_A = LFE(in_channels=self.A_in_c)
#
#        # Attention
#        self.DotProductAttention = ScaledDotProductAttentionOnly(temperature=temperature)
#
#
#    def forward(self, E):
#        #E 端元 A 丰度
#        b, c, h, w = E.size(0), E.size(1), E.size(2), E.size(3)
#        A = torch.ones(b,c,h,w)
#        # Obtaining Values, Keys, and Queries
##        V = self.LFE_A(A)
##        K = self.LFE_E(E)
##        Q = self.LFE_E(E)
#        V =A.cuda()
#        K = E
#        Q = E
#
#        # Obtaining T (Transfered HR Features)
#        T = self.DotProductAttention(V, K, Q)
#        return T
# class Transformer(nn.Module):
#     def __init__(self, input_ch, output_ch):
#         super(Transformer,self).__init__()
#         # self.E_in_c = E_in_c  # Number of input channels = Number of output channels
#         # self.A_in_c = A_in_c
#         # Learnable feature extractors (FE-PAN & FE-HSI)
#         # self.LFE_E = LFE(in_channels=self.E_in_c)
#         # self.LFE_A = LFE(in_channels=self.A_in_c)
#         self.net = nn.Sequential(
#             # nn.Conv2d(input_ch, output_ch, 1, 1, 0, bias=False),
#
#             nn.Conv2d(input_ch, output_ch, 1, 1, 0, bias=True),
# #            nn.BatchNorm2d(output_ch),
#             nn.LeakyReLU(negative_slope=0.0)
#         )
#         # Attention
#         # self.DotProductAttention = ScaledDotProductAttentionOnly(temperature=temperature)
#
#
#     def forward(self, x):
# #        return self.net(x).clamp_(0, 1)
#         return self.net(x)

class Transformer(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        # self.E_in_c = E_in_c  # Number of input channels = Number of output channels
        # self.A_in_c = A_in_c
        # Learnable feature extractors (FE-PAN & FE-HSI)
#        self.LFE_E = LFE(in_channels=self.E_in_c)
#        self.LFE_A = LFE(in_channels=self.A_in_c)

        # Attention
        self.DotProductAttention = ScaledDotProductAttentionOnly(temperature=temperature)


    def forward(self, E,endmember):
        #E 端元 A 丰度
        b, c, h, w = E.size(0), E.size(1), E.size(2), E.size(3)
        # A = torch.ones(b,c)
        # torch.triu(torch.ones(5, 5), diagonal=1)
        # A =torch.triu(A,diagonal=1).view(b,c,h,w).cuda()
        # Obtaining Values, Keys, and Queries
#        V = self.LFE_A(A)
#        K = self.LFE_E(E)
#        Q = self.LFE_E(E)
#        V = A
#        K

        # Obtaining T (Transfered HR Features)
#        T = self.DotProductAttention(V, K, Q)
        T = self.DotProductAttention(E, E,endmember)

        return T

if __name__=="__main__":

    # A = torch.randn(1,128,40,40)
    # E = torch.randn(1,128,40,40).cuda()
    #
    # T = Transformer(64,64,1)
    # output = T(E)
    # print(output.size())
    A = torch.randn(128* 128, 1600)
    E = torch.randn(128,128*128)
    s = torch.mm(E,A)
    s = s.view(-1,40,40)
    print(s.size())





#