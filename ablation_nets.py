import math
import torch
from torch import nn
# import torch.nn.functional as F
# from inspect import isfunction
# import torchvision
# # region denoise model
# import os.path
from generic_blocks import DoubleConv_res, Down, Up, OutConv, FullyConnected

class UNet_generic(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, embed_size=64, bilinear=False, posenc=5, time_concat=True,
                 time_encoder=False):
        print('UNet_Generic')
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.time_encoder = time_encoder
        add_cahnnels_input = 0
        #freqs = torch.linspace(1,magnitude, omega_size)  # torch.rand(omega_size) * magnitude
        self.posenc = posenc
        self.time_concat = time_concat
        if posenc > 0:
            omega_size = self.posenc
            magnitude = 20
            freqs = torch.exp(math.log(magnitude) * torch.linspace(0, 1, omega_size))
            self.omega = nn.Parameter(freqs)
            self.omega.requires_grad = False
            self.positional_encoding = None
            add_cahnnels_input += omega_size * 4
        if time_concat:
            add_cahnnels_input += 1


        self.smart_pad = False
        padfirst = 0 if self.smart_pad else 1
        #self.style_encoder = FullyConnected(n_style, W_SIZE, layers=6)
        #self.padding = padding
        self.input_channels = n_channels + add_cahnnels_input
        t_embed_encoder = embed_size if time_encoder else None
        self.inc = DoubleConv_res(self.input_channels, 64, t_embed=t_embed_encoder, padfirst=padfirst)
        self.down1 = Down(64, 128, t_embed=t_embed_encoder)
        self.down2 = Down(128, 256, t_embed=t_embed_encoder)
        self.down3 = Down(256, 512, t_embed=t_embed_encoder)
        factor = 2 if bilinear else 1 # ==True
        self.down4 = Down(512, 1024 // factor, t_embed=t_embed_encoder)
        self.up1 = Up(1024, 512 // factor, bilinear, t_embed=embed_size)
        self.up2 = Up(512, 256 // factor, bilinear, t_embed=embed_size)
        self.up3 = Up(256, 128 // factor, bilinear, t_embed=embed_size)
        self.up4 = Up(128, 64, bilinear, t_embed=embed_size)
        self.outc = OutConv(64, n_classes)

        self.embd_t_linear = FullyConnected(1, embed_size, layers_num=2)


        #self.norm1 = nn.BatchNorm1d(embed_size)  # , affine=True)

    def forward(self, inputx, input_t):

        t_embed = self.embd_t_linear(input_t)

        input_cat = [inputx]
        if self.posenc > 0:
            encoding = self.get_encoding(inputx)
            input_cat.append(encoding)
        if self.time_concat:
            N, C, H, W = inputx.shape
            t_channel = input_t[..., None, None].expand(N, 1, H, W)
            input_cat.append(t_channel)

        x = torch.cat(input_cat, dim=1)

        x1 = self.inc(x, t_embed)
        x2 = self.down1(x1, t_embed)
        x3 = self.down2(x2, t_embed)
        x4 = self.down3(x3, t_embed)
        x5 = self.down4(x4, t_embed)
        x = self.up1(x5, x4, t_embed)
        x = self.up2(x, x3, t_embed)
        del x4,x5,x3
        x = self.up3(x, x2, t_embed)
        del x2
        x = self.up4(x, x1, t_embed)
        out = self.outc(x)
        #inputx_c = self.skipc(inputx)
        return out+inputx

    def encode(self, shp):
        device = self.omega.device
        B, _, H, W = shp
        row = torch.arange(H, device=device) / H
        enc_row1 = torch.sin(self.omega[None, :] * row[:, None])
        enc_row2 = torch.cos(self.omega[None, :] * row[:, None])
        rows = torch.cat([enc_row1.unsqueeze(1).repeat((1, W, 1)), enc_row2.unsqueeze(1).repeat((1, W, 1))], dim=-1)

        col = torch.arange(W, device=device) / W
        enc_col1 = torch.sin(self.omega[None, :] * col[:, None])
        enc_col2 = torch.cos(self.omega[None, :] * col[:, None])
        cols = torch.cat([enc_col1.unsqueeze(0).repeat((H, 1, 1)), enc_col2.unsqueeze(0).repeat((H, 1, 1))], dim=-1)

        encoding = torch.cat([rows, cols], dim=-1)
        encoding = encoding.permute(2, 0, 1).unsqueeze(0).repeat((B, 1, 1, 1))
        return encoding

    def get_encoding(self, x):
        shp1 = x.shape

        singelton = self.positional_encoding is not None\
                    and self.pe_for_shape[0] == shp1[0] and self.pe_for_shape[2:] == shp1[2:]
        if singelton:
            return self.positional_encoding
        #print('compute for ',shp1)
        self.positional_encoding = self.encode_pad(x.shape) if self.smart_pad else self.encode(x.shape)
        self.pe_for_shape = x.shape
        #print('compute pe')
        return self.positional_encoding