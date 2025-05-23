import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
import numbers
from model.LTE import Conv3DBlock

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def findrate(chanHS, chanMS):
    i = 0
    while True:
        if chanMS * 2 ** i <= chanHS < chanMS * 2 ** (i + 1):
            return i
        i += 1


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # x = x + self.attn(self.norm1(x))
        # x = x + self.ffn(self.norm2(x))
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class CNNBlock(nn.Module):
    def __init__(self,chan_in, n_feat):
        super(CNNBlock, self).__init__()
        # self.conv3D = Conv3DBlock(1,1)
        self.body = nn.Sequential(
            nn.Conv2d(chan_in, n_feat, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=True),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.body(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, in_channel,n_feat):
        super(Downsample, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channel, n_feat, kernel_size=2, stride=2, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU()
        )
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.up(x)
        return x + self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_channel,n_feat):
        super(Upsample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channel, n_feat, kernel_size=2, stride=2, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU()
        )
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.up(x)
        return x+self.body(x)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class Outproj(nn.Module):
    """
        double conv
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.Doubleconv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return self.Doubleconv(x)


class MSVT(nn.Module):
    """GSTB:
        Args: chan_in: image channel nums
    """
    def __init__(self, chan_in,n_feat, heads=2,ffn_expansion_factor=2, LayerNorm_type='with_bias', bias=False,mode=None):
        super().__init__()
        self.mode = mode
        self.encoder = CNNBlock(chan_in=chan_in,n_feat=n_feat)
        # self.encoder = nn.Sequential(*[
        #     TransformerBlock(dim=chan_in, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type)])
        if self.mode == 'up':
            self.sample = Upsample(chan_in,n_feat)
        elif self.mode == 'down':
            self.sample = Downsample(chan_in,n_feat)
        else:
            self.sample = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            # nn.PReLU(),
            # nn.BatchNorm2d(ch_out)
        )

    def forward(self, img):
        return self.sample(self.encoder(img))


class Registration(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels,
                            padding=1),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            # torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            # torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.LeakyReLU(0.2)
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            # torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
            # torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            # # torch.nn.BatchNorm2d(out_channels),
            # torch.nn.LeakyReLU(0.2)
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(Registration, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        # mid_channel = 128
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
            # torch.nn.BatchNorm2d(mid_channel * 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1),
            # torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2,
                                     padding=1, output_padding=1), #75的时候去掉
            torch.nn.LeakyReLU(0.2)
        )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """

        if crop:
            c = (upsampled.size()[2] - bypass.size()[2]) // 2
            upsampled = F.pad(upsampled, (-c, 0, -c, 0))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        bottleneck1 = self.bottleneck(encode_pool3)
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)
        final_layer = final_layer.clamp(min=-5, max=5)
        #75*75
        # encode_block1 = self.conv_encode1(x)
        # # encode_pool1 = self.conv_maxpool1(encode_block1)
        # encode_block2 = self.conv_encode2(encode_block1)
        # # encode_pool2 = self.conv_maxpool2(encode_block2)
        # encode_block3 = self.conv_encode3(encode_block2)
        # # encode_pool3 = self.conv_maxpool3(encode_block3)
        # bottleneck1 = self.bottleneck(encode_block3)
        # decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=False)
        # cat_layer2 = self.conv_decode3(decode_block3)
        # decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        # cat_layer1 = self.conv_decode2(decode_block2)
        # decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        # final_layer = self.final_layer(decode_block1)
        # final_layer = final_layer.clamp(min=-5, max=5)
        ####
        # encode_block1 = self.conv_encode1(x)
        # encode_pool1 = self.conv_maxpool1(encode_block1)
        # encode_block2 = self.conv_encode2(encode_pool1)
        # encode_pool2 = self.conv_maxpool2(encode_block2)
        # # encode_block3 = self.conv_encode3(encode_pool2)
        # # encode_pool3 = self.conv_maxpool3(encode_block3)
        # bottleneck1 = self.bottleneck(encode_pool2)
        # decode_block3 = self.crop_and_concat(bottleneck1, encode_block2, crop=True)
        # cat_layer2 = self.conv_decode3(decode_block3)
        # decode_block2 = self.crop_and_concat(cat_layer2, encode_block1)
        # # cat_layer1 = self.conv_decode2(decode_block2)
        # # decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        # final_layer = self.final_layer(decode_block2)
        # final_layer = final_layer.clamp(min=-5, max=5)
        return final_layer


"""
    Resampling Module
"""
class SpatialTransformation(nn.Module):
    def __init__(self, device='cuda:0'):
        super(SpatialTransformation, self).__init__()
        self.device = device

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]),
                           torch.transpose(torch.unsqueeze(torch.linspace(0.0, width - 1.0, width), 1), 1, 0)).to(
            self.device)
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width])).to(
            self.device)

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)

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
        base = self.repeat(torch.arange(0, batch_size) * dim1, out_height * out_width).to(self.device)

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


class DefomableBlock(nn.Module):
    """
        MMRD
        Args: chanHS: HSI channel nums
              chanMS: MSI channel nums
    """
    def __init__(self, chanHS, chanMS):
        super(DefomableBlock, self).__init__()
        # self.conv3D= Conv3DBlock(1,1)
        self.rate = findrate(chanHS, chanMS)
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chanMS * 2 ** i, chanMS * 2 ** (i + 1), kernel_size=3, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(chanMS * 2 ** (i + 1))
            ) for i in range(self.rate)
        ])
        self.conv.append(nn.Sequential(
            nn.Conv2d(chanMS * 2 ** self.rate, chanHS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(chanHS)
        ))
        self.unet = Registration(chanHS, 2)
        self.spatial_transform = SpatialTransformation()
        self.to_q = nn.Conv2d(chanHS, chanHS, kernel_size=3,padding=1)
        self.to_k = nn.Conv2d(chanHS, chanHS, kernel_size=3,padding=1)
        self.to_v = nn.Conv2d(chanHS, chanHS, kernel_size=3,padding=1)
        # self.to_q = nn.Conv2d(chanHS, chanHS, kernel_size=1,padding=0)
        # self.to_k = nn.Conv2d(chanHS, chanHS, kernel_size=1,padding=0)
        # self.to_v = nn.Conv2d(chanHS, chanHS, kernel_size=1,padding=0)
        # self.to_q = Conv3DBlock(1,1)
        # self.to_k = Conv3DBlock(1,1)
        # self.to_v = Conv3DBlock(1,1)
        # self.replace_conv = nn.Conv2d(chanHS*2, chanHS, kernel_size=3,padding=1) #用于消融实验
    def forward(self, moving_image, fixed_image):
        # for b in self.conv:
        #     fixed_image = b(fixed_image)
        q = self.to_q(fixed_image)
        k = self.to_k(moving_image)
        v = self.to_v(moving_image)

        spatial_attention = torch.matmul(q, k.transpose(-1, -2))
        x = torch.matmul(spatial_attention,v)
        # x = self.replace_conv(torch.cat([moving_image,fixed_image], dim=1))
        deformation_matrix = self.unet(x).permute(0, 2, 3, 1) #b,h,w,2
        registered_image = self.spatial_transform(moving_image.permute(0, 2, 3, 1), deformation_matrix).permute(0, 3, 1,
                                                                                                                2)
        return registered_image

class Backbone(nn.Module):
    """ model
    Args:
        chanHS: HSI channel nums
        chanMS: MSI channel nums
    """
    def __init__(self, chanHS, chanMS, n_feat):
        super().__init__()
        self.chanHS = chanHS
        self.chanMS = chanMS

        self.HSen1 = MSVT(chan_in=chanHS,n_feat=n_feat)
        self.HSen2 = MSVT(chan_in=n_feat,n_feat=n_feat, mode='up')
        self.HSen3 = MSVT(chan_in=n_feat,n_feat=n_feat, mode='up')
        self.HSen4 = MSVT(chan_in=n_feat,n_feat=n_feat, mode='up')

        self.MSen1 = MSVT(chan_in=chanMS,n_feat=n_feat)
        self.MSen2 = MSVT(chan_in=n_feat,n_feat=n_feat, mode='down')
        self.MSen3 = MSVT(chan_in=n_feat,n_feat=n_feat, mode='down')
        self.MSen4 = MSVT(chan_in=n_feat,n_feat=n_feat, mode='down')

        self.d1 = DefomableBlock(chanMS=n_feat, chanHS=n_feat)
        self.d2 = DefomableBlock(chanMS=n_feat, chanHS=n_feat)
        self.d3 = DefomableBlock(chanMS=n_feat, chanHS=n_feat)
        self.d4 = DefomableBlock(chanMS=n_feat, chanHS=n_feat)

        # self.hr1 = Hrnet1(chanHS=chanHS, chanMS=chanMS)
        # self.hr2 = Hrnet2(chanHS=chanHS, chanMS=chanMS)
        # self.hr3 = Hrnet3(chanHS=chanHS, chanMS=chanMS)
        #
        # self.connect1 = Outproj(ch_in=chanHS + chanMS, ch_out=chanHS)
        # self.connect2 = Outproj(ch_in=chanHS + chanMS, ch_out=chanHS)
        # self.connect3 = Outproj(ch_in=chanHS + chanMS, ch_out=chanHS)
        # self.connect4 = Outproj(ch_in=chanHS + chanMS, ch_out=chanHS)
        #
        # self.Recon = nn.Sequential(
        #     nn.Conv2d(chanHS + chanMS + chanHS, chanHS * 2, kernel_size=3, padding=1),
        #     nn.PReLU(),
        #     nn.Conv2d(chanHS * 2, chanHS, kernel_size=3, padding=1),
        #     nn.PReLU(),
        #     nn.Conv2d(chanHS, chanHS, kernel_size=3, padding=1),
        #     nn.Tanh(),
        # )
        # self.Res = nn.Conv2d(chanHS, chanHS, kernel_size=3, padding=1)
        # self.panet = PANet(in_channels=n_feat, out_channels=n_feat)
    def forward(self, x, y):
        x1 = self.HSen1(x)
        del x
        x2 = self.HSen2(x1)
        x3 = self.HSen3(x2)
        x4 = self.HSen4(x3)

        y1 = self.MSen1(y)
        del y
        y2 = self.MSen2(y1)
        y3 = self.MSen3(y2)
        y4 = self.MSen4(y3)

        d1 = self.d1(x1, y4)
        d2 = self.d2(x2, y3)
        d3 = self.d3(x3, y2)
        d4 = self.d4(x4, y1) #转置后的lrhsi
#75*75
        # d1 = self.d1(x1, y3)
        # d2 = self.d2(x2, y2)
        # d3 = self.d3(x3, y1)
        # d4 = self.d4(x4, y1) #转置后的lrhsi
        # scale1 = self.connect1(torch.cat([d1, y4], dim=1)) #尺寸最小的一层
        # scale2 = self.connect2(torch.cat([d2, y3], dim=1))
        # scale3 = self.connect3(torch.cat([d3, y2], dim=1))
        # scale4 = self.connect4(torch.cat([d4, y1], dim=1))
        # x2, x3, x4 = self.hr1(scale1, scale2, scale3, scale4, torch.cat([d1, y4], dim=1))
        # x2, x3 = self.hr2(x2, x3, x4, torch.cat([d2, y3], dim=1))
        # x4 = self.hr3(x2, x3, torch.cat([d3, y2], dim=1))
        # res = self.Recon(torch.cat([x4, torch.cat([d4, y1], dim=1)], dim=1))

        # return res + self.Res(res)
        # x1 = self.panet(x1,x2,x3,x4)
        return x1,y1,d1,d2,d3,d4
        # return x1, y1, d1, d2, d3
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (5,3), 'kernel_size must be 1,3 '
        padding = 2 if kernel_size ==5 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
                                   # nn.ReLU(),
                                   # nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class PANet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PANet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        # self.conv1 =  nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,bias=False),
        #     # nn.BatchNorm2d(n_feat),
        #     nn.LeakyReLU(0.0),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     # nn.BatchNorm2d(n_feat),
        #     nn.LeakyReLU(0.0)
        # )
        # self.conv2 =nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        #     # nn.BatchNorm2d(n_feat),m
        #     nn.LeakyReLU(0.0),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     # nn.BatchNorm2d(n_feat),
        #     nn.LeakyReLU(0.0)
        # )
        # self.conv3 =nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        #     # nn.BatchNorm2d(n_feat),
        #     nn.LeakyReLU(0.0),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     # nn.BatchNorm2d(n_feat),
        #     nn.LeakyReLU(0.0)
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        #     # nn.BatchNorm2d(n_feat),
        #     nn.LeakyReLU(0.0),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     # nn.BatchNorm2d(n_feat),
        #     nn.LeakyReLU(0.0)
        # )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU()
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False),
            # nn.BatchNorm2d(n_feat),
            nn.LeakyReLU()
        )
        # 定义可学习的权重参数
        self.attention0 = SpatialAttention(kernel_size=5)
        self.attention1 = SpatialAttention(kernel_size=5)
        self.attention2 = SpatialAttention(kernel_size=5)
        # self.attention0 = SpatialAttention(kernel_size=3)
        # self.attention1 = SpatialAttention(kernel_size=3)
        # self.attention2 = SpatialAttention(kernel_size=3)
        # self.attention3 = SpatialAttention(kernel_size=3)
        # self.weight1 = nn.Parameter(torch.rand(1))
        # self.weight2 = nn.Parameter(torch.rand(1))
        # self.weight3 = nn.Parameter(torch.rand(1))
        # self.weight4 = nn.Parameter(torch.rand(1))
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0) #3 1
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.conv7 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.conv8 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
    def forward(self,ori_lr_hsi, d1,d2,d3,d4):
    # def forward(self, ori_lr_hsi, d1, d2, d3):
        # Bottom-up pathway
        # x = d1
        x1 = F.relu(self.conv1(d1))
        x2 = F.relu(self.conv2(d2))
        x3 = F.relu(self.conv3(d3))
        x4 = F.relu(self.conv4(d4))
        #
        # # Top-down pathway
        # x3_td = self.downsample(F.relu(self.conv5(x4))) + x3
        # x2_td = self.downsample(F.relu(self.conv6(x3_td))) + x2
        # x1_td = self.downsample(F.relu(self.conv7(x2_td))) + x1

        # x3_td = self.downsample(F.relu(x4)) + x3
        # x2_td = self.downsample(F.relu(x3_td)) + x2
        # x1_td = self.downsample(F.relu(x2_td)) + x1

        x3_td = self.downsample(F.relu(self.conv5(x4))) + x3
        x2_td = self.downsample(F.relu(self.conv6(x3_td))) + x2
        x1_td = self.downsample(F.relu(self.conv7(x2_td))) + x1
        weight0 = self.attention1(x1_td)

        # x3_td = self.downsample(F.relu(self.conv5(x4))) + x3
        # x3_td_d = self.downsample(F.relu(self.conv6(x3_td)))
        # weight2 = self.attention2(x3_td_d)
        # x2_td =  weight2*x3_td_d +x2
        # x2_td_d = self.downsample(F.relu(self.conv7(x2_td)))
        # weight1 = self.attention1(x2_td_d)
        # x1_td = weight1*x2_td_d  + x1
        # weight0 = self.attention0(x1_td)
        return ori_lr_hsi + weight0 *x1_td
        # return ori_lr_hsi+x1_td

def main():
    # Input/Output Testing
    device = 'cuda:0'
    x = torch.randn(1, 102, 32, 32).to(device)
    y = torch.randn(1, 4, 256, 256).to(device)

    model = Backbone(chanHS=102, chanMS=4, n_feat=102).to(device)
    # # print(model(x, y).shape)
    # msvt = MSVT(chan_in=32,n_feat=32,mode="up").to(device)

    # model = Backbone(chanHS=32, chanMS=4).to(device)
    # print(model(x, y).shape)
    x1, x2, out1, out2, out3, out4 =model(x,y)
    # print(out1.shape)
    # print(out2.shape)
    # print(out3.shape)
    pnet = PANet(in_channels=102,out_channels=102)
    pnet.to(device)
    out = pnet(out1,out2,out3,out4)
    print(out.shape)

if __name__ == "__main__":
    main()
