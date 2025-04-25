import math
import time
import options, data
# from model.graph import ICSTN
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
from model.Transformer import Transformer
# from model.LTE import Conv3DBlock
from model.model_multiscale0 import Backbone,PANet

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   factor=opt.lr_decay_gamma,
                                                   patience=opt.lr_decay_patience)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)

    return net


class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one

        return target_tensor.expand_as(input)

    def __call__(self, input):
        #        print(input)
        if(len(input)==1):
            input = torch.sum(input[0], 1)
        else:
            _,c,h,w = input[0].size()
            input[1] = input[1].view(1,-1,h,w)
            input = torch.cat((torch.sum(input[0],1),torch.sum(input[1],0)),dim=0)

        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor)

        return loss

class LossPixlwise(nn.Module):
    def __init__(self):
        super(LossPixlwise, self).__init__()
        self.loss = torch.nn.L1Loss(size_average=False)
    def __call__(self, input,target):
        loss = self.loss(input, target)*1000
        return loss
def kl_divergence(p, q):
    p = F.softmax(p)
    q = F.softmax(q)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()
        self.register_buffer('zero', torch.tensor(0.01, dtype=torch.float))

    def __call__(self, input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss


def my_define_msi_1(input_ch, gpu_ids, ngf, init_type='kaiming', init_gain=0.02):
    net = my_Msi_1(input_c=input_ch, ngf=ngf)
    return init_net(net, init_type, init_gain, gpu_ids)


#
# def my_transformer():
#     return init_net()kenen
class my_Msi_1(nn.Module):
    def __init__(self, input_c, ngf):
        super(my_Msi_1, self).__init__()
        # self.net3d = Conv3DBlock(in_channels=1,mid_channels=1)
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf * 2, 5, 1, 2, padding_mode='zeros'),  # 真实 5.1 2
            # nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True),  # 真实 0.0
            nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1, padding_mode='zeros'),
            nn.LeakyReLU(0.2, True),  # 真实 0.0
            # nn.Conv2d(ngf*2 , ngf*2, 3, 1, 1, padding_mode='zeros') ,# 311
            # # nn.BatchNorm2d(ngf*4),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(ngf * 2, ngf * 2, 1, 1, 0, padding_mode='zeros'),  # 311
            # # nn.BatchNorm2d(ngf*4),
            # nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, 1, 1, 0, padding_mode='zeros'),
            # nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        # return self.net3d(self.net(x))
        return self.net(x)


def my_define_msi_2(output_ch, gpu_ids, ngf, init_type='kaiming', init_gain=0.02, useSoftmax='Yes'):
    net = my_Msi_2(output_c=output_ch, ngf=ngf, useSoftmax=useSoftmax)
    return init_net(net, init_type, init_gain, gpu_ids)


class my_Msi_2(nn.Module):
    def __init__(self, output_c, ngf, useSoftmax='Yes'):
        super(my_Msi_2, self).__init__()
        self.net3d = Conv3DBlock(in_channels=1, mid_channels=1)
        self.net1 = nn.Sequential(
            nn.Conv2d(ngf * 4, output_c, 1, 1, 0),
            # nn.LeakyReLU(0.2),
            #  nn.Conv2d(output_c*4, output_c * 2, 3, 1, 1),
            #  nn.LeakyReLU(0.2),
            # nn.Conv2d(output_c*2, output_c, 1, 1, 0),
            # nn.LeakyReLU(0.2)
        )
        # self.net2 = nn.Sequential(
        #    nn.Conv2d(ngf*16, output_c, 1, 1, 0),
        #    nn.LeakyReLU(0.0)
        # )
        self.usesoftmax = useSoftmax
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # if self.usesoftmax == "Yes":
        #     return self.softmax(self.net(x))
        # elif self.usesoftmax == 'No':
        #     return self.net(x).clamp_(0,1)
        # params = list(self.net1.named_parameters())[0][1].cuda()
        if self.usesoftmax == "Yes":
            return self.softmax(self.net1(self.net3d(x)))
        elif self.usesoftmax == 'No':
            return self.net1(self.net3d(x)).clamp_(0, 1)


def my_define_lr_1(input_ch, gpu_ids, ngf, init_type='kaiming', init_gain=0.02):
    net = my_Lr_1(input_c=input_ch, ngf=ngf)
    return init_net(net, init_type, init_gain, gpu_ids)


class my_Lr_1(nn.Module):
    def __init__(self, input_c, ngf):
        super(my_Lr_1, self).__init__()
        self.net3d = Conv3DBlock(in_channels=1, mid_channels=1)
        self.net = nn.Sequential(
            # nn.Conv2d(input_c, ngf*2, 1, 1, 0),
            # nn.LeakyReLU(0.2, True),
            # # nn.Conv2d(ngf*2, ngf * 2, 1, 1, 0),
            # # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(ngf*2, ngf*4, 1, 1, 0),
            # nn.LeakyReLU(0.2, True), #光谱信息回复的很差
            # nn.Conv2d(ngf*4, ngf*4, 1, 1, 0),
            # nn.LeakyReLU(0.2, True)
            nn.Conv2d(input_c, ngf * 2, 3, 1, 1),
            nn.LeakyReLU(0.0, True),
            nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, 1, 1),
            # nn.Conv2d(ngf*4, ngf*4, 1, 1, 0),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.net(self.net3d(x))


def my_define_lr_2(output_ch, gpu_ids, ngf, init_type='kaiming', init_gain=0.02, useSoftmax='Yes'):
    net = my_Lr_2(output_c=output_ch, ngf=ngf, useSoftmax=useSoftmax)
    return init_net(net, init_type, init_gain, gpu_ids)


class my_Lr_2(nn.Module):
    def __init__(self, output_c, ngf, useSoftmax='Yes'):
        super(my_Lr_2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ngf * 4, output_c, 1, 1, 0),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(output_c, output_c, 1, 1, 0),
            # nn.LeakyReLU(0.0),
            # nn.Conv2d(output_c,output_c,1,1,0),
            # nn.LeakyReLU(0.0)
        )
        self.usesoftmax = useSoftmax
        self.softmax = nn.Softmax(dim=0)
        self.net3d = Conv3DBlock(in_channels=1, mid_channels=1, kernal=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x):
        # params = list(self.net.named_parameters())[0][1].cuda()
        if self.usesoftmax == "Yes":
            out = self.net(self.net3d(x))
            return self.softmax(out)
        elif self.usesoftmax == 'No':
            # return self.net(self.net3d(x)).clamp_(0,1)
            return self.net(x).clamp_(0, 1)


def define_s2img(input_ch, output_ch, gpu_ids, init_type='kaiming', init_gain=0.02):
    net = S2Img(input_c=input_ch, output_c=output_ch)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_bm_net(temperature, gpu_ids, init_type='kaiming', init_gain=0.02):
    net = Transformer(temperature)
    return init_net(net, init_type, init_gain, gpu_ids)


class S2Img(nn.Module):
    def __init__(self, input_c, output_c):
        super(S2Img, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).clamp_(0, 1)

def define_s2img_2stream(input_c1, input_tf, output_ch, gpu_ids, init_type='kaiming', init_gain=0.02):
    net = S2Img_2stream(input_c1=input_c1, input_tf=input_tf, output_c=output_ch)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_s2img_2stream_msi(input_c1, input_tf, output_ch, gpu_ids, init_type='kaiming', init_gain=0.02):
    net = S2Img_2stream_msi(input_c1=input_c1, input_tf=input_tf, output_c=output_ch)
    return init_net(net, init_type, init_gain, gpu_ids)
class S2Img_2stream(nn.Module):
    def __init__(self, input_c1, input_tf, output_c):
        super(S2Img_2stream, self).__init__()
        # self.height = height
        # self.weight = weight
        self.band = output_c
        self.net1 = nn.Sequential(
            nn.Conv2d(input_c1, output_c, 1, 1, 0, bias=False),
            # nn.ReLU()
            # nn.Conv2d(output_c, output_c, 1, 1, 0, bias=False),
            # nn.ReLU()
        )
        self.net2 = nn.Sequential(
            # nn.Conv3d(input_c1, output_c,  bias=True),
            #
            # nn.Conv2d(input_c1, input_c1*2, 1, 1, 0, bias=False),
            # nn.ReLU(),
            # nn.Conv2d(input_c1*2, input_c1*2, 1, 1, 0, bias=False),
            # nn.ReLU(),
            # # nn.Linsear(input_c1,height*weight),
            nn.Conv2d(input_c1, input_c1*(input_c1+1)//2, 1, 1, 0, bias=False),
            nn.ReLU()
            # nn.Conv2d(output_c, output_c, 1, 1, 0, bias=True)
        )
        # self.para = list(self.net1.named_parameters())[0][1]
        # self.net2 = Endmenber_CNN(self.band)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.tf = input_tf

    def forward(self, x1,abund_bmm):
        out1 = self.net1(x1)
        para = self.relu(list(self.net1.named_parameters())[0][1])
        time1 = time.time()
        endmember_bmm = self.tf(para, True)
        band, ch = endmember_bmm.size(0), endmember_bmm.size(1)  # ch 上三角元素个数 M+1 * M/2
        endmember_bmm = self.relu(endmember_bmm.view(band, ch))
        # abund_bmm = self.tf(x1.permute(2,3,1,0),False) #x1 1,64,32,32
        # abund_bmm = self.relu(self.tf(x1, False))
        if abund_bmm is None:
            abund_bmm = self.net2(x1)
            c, h, w = abund_bmm.size(1), abund_bmm.size(2), abund_bmm.size(3)
            abund_bmm = abund_bmm.view(c, h * w)
        else:
            c, x = abund_bmm.size(0), abund_bmm.size(1)
            h = int(math.sqrt(x))
            w = h
        abund_bmm = self.softmax(abund_bmm)
        # tff = tff.view(self.band, self.band)
        # self.net2 = Endmenber_CNN(tff,self.band)
        # out2=self.net2(tff)
        # out2 = out2.view(1,self.band,self.height,self.weight)
        out2 = torch.mm(endmember_bmm, abund_bmm)
        # out2 = F.softmax(out2, dim=-1)
        # out2 = endmember_bmm
        # out2 = self.sigmoid(out2)
        out2 = out2.view(1, -1, h, w)
        # out3 = out2
        # b_i = self.sigmoid(self.net2(torch.cat([out1,out2],1)).view(1, -1, h*w))

        # b_i = self.sigmoid(self.net2(out3).view(1, -1, h * w))
        # b_i = b_i.view(-1, h, w)
        # out2 = torch.mul(out2, b_i)
        # out2 = out2.view(-1, h, w)
        # print(time.time()-time1)
        return (out1 + out2).clamp_(0, 1),abund_bmm
        # return out1.clamp_(0,1),abund_bmm

class S2Img_2stream_msi(nn.Module):
    def __init__(self, input_c1, input_tf, output_c):
        super(S2Img_2stream_msi, self).__init__()
        # self.height = height
        # self.weight = weight
        self.band = output_c
        self.net1 = nn.Sequential(
            nn.Conv2d(input_c1, output_c, 1, 1, 0, bias=False),
            # nn.ReLU()
            # nn.Conv2d(output_c, output_c, 1, 1, 0, bias=False),
            # nn.ReLU()
        )
        self.net2 = nn.Sequential(
            # nn.Conv3d(input_c1, output_c,  bias=True),
            #
            # nn.Conv2d(output_c, output_c, 3, 1, 1, bias=False),
            # nn.ReLU(),
            # nn.Conv2d(output_c, output_c, 3, 1, 1, bias=True),
            # nn.ReLU(),
            # # nn.Linsear(input_c1,height*weight),
            nn.Conv2d(input_c1, input_c1*(input_c1+1)//2, 1, 1, 0, bias=True),
            nn.ReLU()
            # nn.Conv2d(output_c, output_c, 1, 1, 0, bias=True)
        )
        self.net3 = nn.Sequential(
            nn.Linear(input_c1*(input_c1+1)//2,input_c1*(input_c1+1)),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(input_c1*(input_c1+1),input_c1*(input_c1+1)//2),
            # nn.Dropout(0.1),
            nn.ReLU())
        # self.para = list(self.net1.named_parameters())[0][1]
        # self.net2 = Endmenber_CNN(self.band)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.tf = input_tf

    def forward(self, x1,abund_bmm):
        out1 = self.net1(x1)
        para = self.relu(list(self.net1.named_parameters())[0][1])
        time1 = time.time()
        endmember_bmm = self.tf(para, True)

        band, ch = endmember_bmm.size(0), endmember_bmm.size(1)  # ch 上三角元素个数 M+1 * M/2
        endmember_bmm = self.net3(endmember_bmm.view(band, ch))
        # abund_bmm = self.tf(x1.permute(2,3,1,0),False) #x1 1,64,32,32
        # abund_bmm = self.relu(self.tf(x1, False))
        if abund_bmm is None:
            abund_bmm = self.net2(x1)
            c, h, w = abund_bmm.size(1), abund_bmm.size(2), abund_bmm.size(3)
            abund_bmm = abund_bmm.view(c, h * w)
        else:
            c, x = abund_bmm.size(0), abund_bmm.size(1)
            h = int(math.sqrt(x))
            w = h
        abund_bmm = self.softmax(abund_bmm)
        # tff = tff.view(self.band, self.band)
        # self.net2 = Endmenber_CNN(tff,self.band)
        # out2=self.net2(tff)
        # out2 = out2.view(1,self.band,self.height,self.weight)
        out2 = torch.mm(endmember_bmm, abund_bmm)
        # out2 = F.softmax(out2, dim=-1)
        # out2 = endmember_bmm
        # out2 = self.sigmoid(out2)
        out2 = out2.view(1, -1, h, w)
        # out3 = out2
        # b_i = self.sigmoid(self.net2(torch.cat([out1,out2],1)).view(1, -1, h*w))

        # b_i = self.sigmoid(self.net2(out3).view(1, -1, h * w))
        # b_i = b_i.view(-1, h, w)
        # out2 = torch.mul(out2, b_i)
        # out2 = out2.view(-1, h, w)
        # print(time.time()-time1)
        return (out1 + out2).clamp_(0, 1), abund_bmm
        # return out1.clamp_(0,1),abund_bmm
def define_spectral_AM(input_ch, input_hei, input_wid, gpu_ids, init_type='mean_channel', init_gain=0.02):
    net = spectral_AM(input_c=input_ch, output_c=input_ch, input_h=input_hei, input_w=input_wid)
    return init_net(net, init_type, init_gain, gpu_ids)


class spectral_AM(nn.Module):
    def __init__(self, input_c, output_c, input_h, input_w):
        super(spectral_AM, self).__init__()
        # self.net = nn.Conv2d(input_c, output_c, (input_h, input_w), 1, 0, groups=input_c)
        self.net = nn.Sequential(
            # nn.Conv2d(input_c, input_c, 1, 1, padding=0),
            # nn.LeakyReLU(0.2),
            nn.Conv2d(input_c, output_c, (input_h, input_w), 1, 0, groups=input_c),
        )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.softmax(self.net(x))
        # return self.sigmoid(self.net(x))


def define_spatial_AM(input_ch, kernel_sz, gpu_ids, init_type='mean_space', init_gain=0.02):
    net = spatial_AM(input_c=input_ch, kernel_s=kernel_sz)
    return init_net(net, init_type, init_gain, gpu_ids)


class spatial_AM(nn.Module):
    def __init__(self, input_c, kernel_s):
        super(spatial_AM, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(input_c, input_c, kernel_s, 1, padding=int((kernel_s - 1) / 2)),
                                 nn.LeakyReLU(0.2),
                                 # nn.Conv2d(input_c, input_c, kernel_s, 1, padding=int((kernel_s - 1) / 2)),
                                 # nn.LeakyReLU(0.2),
                                 nn.Conv2d(input_c, 1, kernel_s, 1, padding=int((kernel_s - 1) / 2)),
                                 nn.LeakyReLU(0.2)
                                 )
        # self.net = nn.Conv2d(input_c,1,kernel_s,1,padding=int((kernel_s-1)/2))
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        # self.net3d = Conv3DBlock(in_channels=1, mid_channels=1, kernal=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x):
        b, c, height, width = x.size()
        # x = self.net3d(x)
        # one = torch.ones([1,1,height*width])
        # SAmap =  self.sigmoid(one.cuda() -self.net(x).view(b, -1, height*width))
        # return SAmap.view(b, 1, height, width)
        return self.sigmoid(self.net(x))


def define_Merge(input_c, out_c, gpu_ids, kernal=1, init_type='mean_space', init_gain=0.02):
    net = Mergenet(input_c, out_c, kernal)
    return init_net(net, init_type, init_gain, gpu_ids)


class Mergenet(nn.Module):
    def __init__(self, input_c, out_channels, kernel_s):
        super(Mergenet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, input_c, 1, 1, padding=int((1 - 1) / 2)),
            nn.LeakyReLU(0.0),
        )
        # self.net1 = nn.Conv2d(input_c,out_channels,kernel_s,1,padding=int((kernel_s-1)/2))

        # self.net2 =
        # self.softmax = nn.Softmax(dim=2)
        # self.sigmoid = nn.Sigmoid()
        # self.net3d = Conv3DBlock(in_channels=1, mid_channels=1, kernal=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x1, x2):
        # b, c, height, width = x.size()
        # x = self.net3d(x)
        # output =x1+ self.net(x2)
        output = x1 + x2
        return output.clamp_(0, 1)


def define_Merge_displacementfield(input_c, out_c, gpu_ids, kernal=1, init_type='mean_space', init_gain=0.02):
    net = Mergenet4dsfield(input_c, out_c, kernal)
    return init_net(net, init_type, init_gain, gpu_ids)


class Mergenet4dsfield(nn.Module):
    def __init__(self, input_c, input_h, input_w, output_c):
        super(Mergenet4dsfield, self).__init__()
        # self.net_spectral = nn.Sequential(
        #         nn.Conv2d(input_c, input_c, 1, 1, padding=int((1-1)/2)),
        #         nn.LeakyReLU(0.0),
        #     )

        self.net_spectral = nn.Sequential(
            # nn.Conv2d(input_c, input_c, 1, 1, padding=0),
            # nn.LeakyReLU(0.2),
            nn.Conv2d(input_c, input_c, (input_h, input_w), 1, 0, groups=input_c),
        )
        # self.net1 = nn.Conv2d(input_c,out_channels,kernel_s,1,padding=int((kernel_s-1)/2))
        self.net_spatial = nn.Sequential(
            nn.Conv2d(input_c, input_c, 1, 1, padding=int((1 - 1) / 2)),
            nn.LeakyReLU(0.0),
        )
        self.softmax = nn.Softmax(dim=1)
        # self.net2 =
        # self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        # self.net3d = Conv3DBlock(in_channels=1, mid_channels=1, kernal=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, hsi, msi):
        spe_matrix = self.softmax(self.net_spectral(hsi))
        x1 = torch.mul(msi, spe_matrix)
        spa_matrix = self.sigmoid(self.net_spatial(msi))
        x2 = torch.mul(hsi, spa_matrix)
        hsi = torch.cat((hsi, x1), dim=1)
        msi = torch.cat((msi, x2), dim=1)
        return hsi, msi


def define_psf(scale, gpu_ids, init_type='mean_space', init_gain=0.02):
    net = PSF(scale=scale)
    return init_net(net, init_type, init_gain, gpu_ids)


class PSF(nn.Module):
    def __init__(self, scale):
        super(PSF, self).__init__()
        self.net = nn.Conv2d(1, 1, scale, scale, 0, bias=False)

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.net(x[:, i, :, :].view(batch, 1, height, weight)) for i in range(channel)],
                         1)  # same as groups=input_c, i.e. channelwise conv


def define_psf_2(scale, gpu_ids, init_type='mean_space', init_gain=0.02):
    net = PSF_2(scale=scale)
    return init_net(net, init_type, init_gain, gpu_ids)


class PSF_2(nn.Module):
    def __init__(self, scale):
        super(PSF_2, self).__init__()
        self.net = nn.Conv2d(1, 1, scale, scale, 0, bias=False)
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.net(x[:, i, :, :].view(batch, 1, height, weight)) for i in range(channel)], 1)


def define_hr2msi(args, hsi_channels, msi_channels, sp_matrix, sp_range, gpu_ids, init_type='mean_channel',
                  init_gain=0.02):
    if args.isCalSP == "No":
        net = matrix_dot_hr2msi(sp_matrix)
    elif args.isCalSP == "Yes":
        net = convolution_hr2msi(hsi_channels, msi_channels, sp_range)
    return init_net(net, init_type, init_gain, gpu_ids)


class convolution_hr2msi(nn.Module):
    def __init__(self, hsi_channels, msi_channels, sp_range):
        super(convolution_hr2msi, self).__init__()

        self.sp_range = sp_range.astype(int)
        self.length_of_each_band = self.sp_range[:, 1] - self.sp_range[:, 0] + 1
        self.length_of_each_band = self.length_of_each_band.tolist()

        self.conv2d_list = nn.ModuleList([nn.Conv2d(x, 1, 1, 1, 0, bias=False) for x in self.length_of_each_band])

    def forward(self, input):
        scaled_intput = input
        cat_list = []
        for i, layer in enumerate(self.conv2d_list):
            input_slice = scaled_intput[:, self.sp_range[i, 0]:self.sp_range[i, 1] + 1, :, :]
            out = layer(input_slice).div_(layer.weight.data.sum(dim=1).view(1))
            cat_list.append(out)
        return torch.cat(cat_list, 1).clamp_(0, 1)




class matrix_dot_hr2msi(nn.Module):
    def __init__(self, spectral_response_matrix):
        super(matrix_dot_hr2msi, self).__init__()
        self.register_buffer('sp_matrix', torch.tensor(spectral_response_matrix.transpose(1, 0)).float())

    def __call__(self, x):
        batch, channel_hsi, heigth, width = list(x.size())
        channel_msi_sp, channel_hsi_sp = list(self.sp_matrix.size())
        hmsi = torch.bmm(self.sp_matrix.expand(batch, -1, -1),
                         torch.reshape(x, (batch, channel_hsi, heigth * width))).view(batch, channel_msi_sp, heigth,
                                                                                      width)
        return hmsi


class NonZeroClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0, 1e8)


class ZeroOneClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0, 1)


class SumToOneClipper(object):

    def __call__(self, module):
        if hasattr(module, 'weight'):
            if module.in_channels != 1:
                w = module.weight.data
                w.clamp_(0, 10)
                w.div_(w.sum(dim=1, keepdim=True))
            elif module.in_channels == 1:
                w = module.weight.data
                w.clamp_(0, 5)


def define_ICSTN():
    opt = options.set(training=True)
    pInit = data.genPerturbations(opt)
    pInit = torch.reshape(pInit, [1, 2]).cuda()
    model = ICSTN(opt)
    return model, pInit


# def define_displacementfiled(in_channels, out_channels, gpu_ids, init_type="kaiming", init_gain=0.02):
#     net = UNet(in_channels, out_channels)
#     # import ipdb
#     # ipdb.set_trace()
#     # net = DisplacementField(in_channels, out_channels)
#     return init_net(net, init_type, init_gain, gpu_ids)


def define_spatial_transform(img_size, use_gpu=True):
    net = SpatialTransformation(use_gpu)
    # net = SpatialTransformer(img_size)
    return net
def define_multiscaleNet(hsi_channels,msi_channels,n_feat, gpu_ids, init_type="kaiming", init_gain=0.02, use_gpu=True):
    net = Backbone(hsi_channels, msi_channels, n_feat)
    # net = SpatialTransformer(img_size)
    return init_net(net, init_type, init_gain, gpu_ids)
def define_PANet(in_channels, out_channels, gpu_ids, init_type="kaiming", init_gain=0.02, use_gpu=True):
    net = PANet(in_channels,out_channels)
    return init_net(net, init_type, init_gain,gpu_ids)
class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(
            torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width - 1.0, width), 1), 1, 0),
        )
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]), )

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu == True:
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda()
        return torch.squeeze(torch.reshape(x, (-1, 1)))

    def interpolate(self, im, x, y):

        im = F.pad(im, (0, 0, 1, 1, 1, 1, 0, 0))

        batch_size, height, width, channels = im.shape

        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width * height
        base = self.repeat(torch.arange(0, batch_size) * dim1, out_height * out_width)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1, 0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1, 0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1, 0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1, 0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1, 0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1, 0)
        wb = (dx * (1 - dy)).transpose(1, 0)
        wc = ((1 - dx) * dy).transpose(1, 0)
        wd = ((1 - dx) * (1 - dy)).transpose(1, 0)

        output = torch.sum(torch.squeeze(torch.stack([wa * Ia, wb * Ib, wc * Ic, wd * Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])

        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)


# class UNet(nn.Module):
#     def contracting_block(self, in_channels, out_channels, kernel_size=3):
#         """
#         This function creates one contracting block
#         """
#         block = torch.nn.Sequential(
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1, ),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1, ),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#         )
#         return block
#
#     def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
#         """
#         This function creates one expansive block
#         """
#         block = torch.nn.Sequential(
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1, ),
#             torch.nn.BatchNorm2d(mid_channel),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1, ),
#             torch.nn.BatchNorm2d(mid_channel),
#             torch.nn.ReLU(),
#             # torch.nn.ConvTranspose2d(
#             #     in_channels=mid_channel,
#             #     out_channels=out_channels,
#             #     kernel_size=3,
#             #     stride=2,
#             #     padding=1,
#             #     output_padding=1,
#             # ),
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1, ),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#         )
#         return block
#
#     def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
#         """
#         This returns final block
#         """
#         block = torch.nn.Sequential(
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1, ),
#             torch.nn.BatchNorm2d(mid_channel),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1, ),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.ReLU(),
#         )
#         return block
#
#     def __init__(self, in_channel, out_channel):
#         super(UNet, self).__init__()
#         # Encode
#         self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
#         self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=1)
#         self.conv_encode2 = self.contracting_block(32, 64)
#         self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv_encode3 = self.contracting_block(64, 128)
#         # self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
#         # Bottleneck
#         mid_channel = 128
#         #原始
#         self.bottleneck = torch.nn.Sequential(
#             torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1, ),
#             torch.nn.BatchNorm2d(mid_channel * 2),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1, ),
#             torch.nn.BatchNorm2d(mid_channel),
#             torch.nn.ReLU(),
#             torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
#             # torch.nn.BatchNorm2d(mid_channel),
#             torch.nn.ReLU(),
#         )
#         # self.bottleneck = torch.nn.Sequential(
#         #     torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1, ),
#         #     torch.nn.BatchNorm2d(mid_channel * 2),
#         #     torch.nn.LeakyReLU(0.1),
#         #     torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1, ),
#         #     torch.nn.BatchNorm2d(mid_channel),
#         #     torch.nn.LeakyReLU(0.1),
#         #     torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
#         #     # torch.nn.BatchNorm2d(mid_channel),
#         #     torch.nn.LeakyReLU(0.0),
#         # )
#         # Decode
#         self.conv_decode3 = self.expansive_block(256, 128, 64)
#         self.conv_decode2 = self.expansive_block(128, 64, 32)
#         self.final_layer = self.final_block(64, 32, out_channel)
#
#     def crop_and_concat(self, upsampled, bypass, resize=True):
#         """
#         This layer resize the layer from contraction block and concat it with expansive block vector
#         """
#         if resize:
#             # c = (bypass.size()[2] - upsampled.size()[2]) // 2
#             # bypass = F.pad(bypass, (-c, -c, -c, -c))
#             bypass = nn.functional.interpolate(bypass, upsampled.size()[2:], mode="bilinear", align_corners=False)
#         return torch.cat((upsampled, bypass), 1)
#
#     def forward(self, x):
#         # Encode
#         encode_block1 = self.conv_encode1(x)
#         # encode_pool1 = self.conv_maxpool1(encode_block1)
#         encode_block2 = self.conv_encode2(encode_block1)
#         # encode_pool2 = self.conv_maxpool2(encode_block2)
#         encode_block3 = self.conv_encode3(encode_block2)
#         # encode_pool3 = self.conv_maxpool3(encode_block3)
#         # Bottleneck
#         bottleneck1 = self.bottleneck(encode_block3)
#         # Decode
#         decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
#         cat_layer2 = self.conv_decode3(decode_block3)
#         decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
#         cat_layer1 = self.conv_decode2(decode_block2)
#         decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
#         # decode_block1 = nn.functional.interpolate(decode_block1, x.size()[2:], mode="bilinear", align_corners=False)
#         final_layer = self.final_layer(decode_block1)
#
#         # import ipdb
#         # ipdb.set_trace()
#         return final_layer


if __name__ == '__main__':
    hsi_channels=102
    msi_channels=4
    device = 'cuda:0'
    n_feat = 32
    x = torch.randn(1, 102, 32, 32).to(device)
    y = torch.randn(1, 4, 256, 256).to(device)
    gpuids= ['cuda:0']
    model=define_multiscaleNet(hsi_channels, msi_channels, n_feat, gpuids)
    output = model(x,y)
    print(output[0].shape)