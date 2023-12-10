import math
import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv_res_0(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, t_embed=50, padfirst=1):
        super().__init__()
        #print('dc, 0', end=' ')
        if not mid_channels:
            mid_channels = out_channels
        self.conv0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padfirst, bias=False)
        self.norm0 = nn.InstanceNorm2d(mid_channels, affine=(t_embed is None))#nn.BatchNorm2d(mid_channels)
        self.activation0 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.conv1 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=(t_embed is None))#nn.BatchNorm2d(out_channels)
        self.activation1 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.padfirst = padfirst
        self.t_embed = t_embed
        if t_embed is not None:
            self.linear_t0gamma = FullyConnected(t_embed, mid_channels, layers_num=2, act_last=False)#nn.Linear(t_embed, mid_channels)
            self.linear_t0betta = FullyConnected(t_embed, mid_channels, layers_num=2, act_last=False)
            self.linear_t1gamma = FullyConnected(t_embed, out_channels, layers_num=2, act_last=False)
            self.linear_t1betta = FullyConnected(t_embed, out_channels, layers_num=2, act_last=False)

    def forward(self, x, t):
        id = self.skip(x) if self.padfirst==1 else self.skip(x[:,:,1:-1,1:-1])
        x = self.conv0(x)
        x = self.norm0(x)
        if self.t_embed is not None:
            t0g = self.linear_t0gamma(t)
            t0b = self.linear_t0betta(t)
            t1g = self.linear_t1gamma(t)
            t1b = self.linear_t1betta(t)
            #print('CODA MEM INMODEL:', [torch.cuda.memory_allocated()])
            x = x * t0g[..., None, None] + t0b[..., None, None]
        x = self.activation0(x+id)
        id2 = x
        x = self.conv1(x)
        x = self.norm1(x)
        if self.t_embed is not None:
            x = x * t1g[..., None, None] + t1b[..., None, None]
        x = self.activation1(x+id+id2)
        return x

class DoubleConv_res_changes(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, t_embed=50):
        super().__init__()
        print('dc,changes', end= ' ')
        if not mid_channels:
            mid_channels = out_channels
        self.conv0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm0 = nn.InstanceNorm2d(mid_channels, affine=(t_embed is None))#nn.BatchNorm2d(mid_channels)

        self.activation0 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.conv1 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=(t_embed is None))#nn.BatchNorm2d(out_channels)
        self.activation1 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.t_embed = t_embed
        if t_embed is None:
            self.norm0 = torch.nn.BatchNorm2d(mid_channels)
            self.norm1 = torch.nn.BatchNorm2d(out_channels)
        if t_embed is not None:
            self.linear_t0gamma = FullyConnected(t_embed, mid_channels, layers_num=2, act_last=False)#nn.Linear(t_embed, mid_channels)
            self.linear_t0betta = FullyConnected(t_embed, mid_channels, layers_num=2, act_last=False)
            self.linear_t1gamma = FullyConnected(t_embed, out_channels, layers_num=2, act_last=False)
            self.linear_t1betta = FullyConnected(t_embed, out_channels, layers_num=2, act_last=False)
        #self.linear_id_bias = FullyConnected(64, out_channels, layers_num=2, act_last=False)

    def forward(self, x, t):
        id = self.skip(x) #+ self.linear_id_bias(t)[..., None, None]
        x = self.conv0(x)
        x = self.norm0(x)
        if self.t_embed is not None:
            t0g = self.linear_t0gamma(t)
            t0b = self.linear_t0betta(t)
            t1g = self.linear_t1gamma(t)
            t1b = self.linear_t1betta(t)
            x = x * t0g[..., None, None] + t0b[..., None, None]
        x = self.activation0(x+id)
        id2 = x
        x = self.conv1(x)
        x = self.norm1(x)
        if self.t_embed is not None:
            x = x * t1g[..., None, None] + t1b[..., None, None]
        x = self.activation1(x+id+id2)
        return x
DoubleConv_res = DoubleConv_res_0#changes

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, t_embed=None):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv_res(in_channels, out_channels, t_embed=t_embed)

    def forward(self, x, t):
        x = self.maxpool(x)
        x = self.conv(x, t)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, t_embed=None):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear: # ==True
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_res(in_channels, out_channels, t_embed=t_embed)
            '''
            elif bilinear=='shuffle':
                self.up = suffle_upsample(scale_factor=2, in_channels=in_channels, out_channels = in_channels // 2)
                self.conv = DoubleConv_res(in_channels, out_channels, t_embed=t_embed)
            '''
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_res(in_channels, out_channels, t_embed=t_embed)


    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FullyConnected(nn.Module):
    def __init__(self, in_feat, out_feat, layers_num=1, act_last=True):
        super(FullyConnected, self).__init__()
        layers_list = []
        assert layers_num > 0
        for i in range(layers_num):
            layers_list.append(nn.Linear(in_feat, out_feat))
            in_feat = out_feat
            act = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            layers_list.append(act)
        if not act_last:
            layers_list = layers_list[:-1]
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layers(x)
