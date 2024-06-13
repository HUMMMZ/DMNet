import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from functools import partial
from segmentation_models_pytorch import create_model
from .vmamba import VSSM
import segmentation_models_pytorch as smp

nonlinearity = partial(F.relu, inplace=True)

BN_EPS = 1e-4  #1e-4  #1e-5


class H_Net(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        #self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        #self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        #self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        #self.conv1 = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.down2 = VSSM(in_chans=32,
                           num_classes=1,
                           depths=[2,2,2,2],
                           depths_decoder=[2,2,2,1],
                           drop_path_rate=0.2,
                        )  # 512
        #self.down2 = VSSM(64)  # 256
        #self.down3 = VSSM(128)  # 128
        #self.down4 = VSSM(256)  # 64

        # ce_net encoder part
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock137(512)  # 空洞卷积
        self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.e4_up = nn.ConvTranspose2d(516, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        # the CAC block
        self.CAC = DACblock137(256)
        self.CAC_Ce = DACblock137(512)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d4_conv = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1, bias=True)
        self.d3_conv = nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.d2_conv = nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1, bias=True)
        #self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)
        
        
        # the encoder of Left_VSSBlock
        


    def forward(self, x):

        # L-Encoder Part
        l_x = x                     #[3, 256, 256]
        out0 = self.down1(l_x) # conv1 [32,256,256]        
        out4, skip_list = self.down2(out0) # conv2 [64,128,128]
        out1 = skip_list[0]
        out2 = self.d2_conv(skip_list[1])    #[128, 64, 64]-->[64, 64, 64]
        out3 = self.d3_conv(skip_list[2])    #[256, 32, 32]-->[128, 32, 32]
        out4 = self.d4_conv(out4.permute(0, 3, 1, 2))    #[512, 16, 16]-->[256, 16, 16]

        # R-Encoder part
        rx = x                      #[3, 256, 256]
        e0 = self.firstconv(rx)     #[64,128,128]
        e0 = self.firstbn(e0)       #[64,128,128]
        e0 = self.firstrelu(e0)     #[64, 128, 128]
        pe0 = self.firstmaxpool(e0) #[64, 64, 64]
        e1 = self.encoder1(pe0)     #[64, 64, 64]
        e2 = self.encoder2(e1)      #[128, 32, 32]
        e3 = self.encoder3(e2)      #[256, 16, 16]
        e4 = self.encoder4(e3)      #[512, 8, 8]
        #print(e0.shape, pe0.shape, e1.shape, e2.shape, e3.shape, e4.shape)

        # Center of CE_Net
        e4 = self.CAC_Ce(e4)        #[512, 8, 8]
        # e4 = self.dblock(e4)
        e4 = self.spp(e4)        #[512, 8, 8]
        # the center part
        e4_up = self.e4_up(e4)     #[256, 16, 16]
        CAC_out = self.CAC(out4)   #[256, 16, 16]
        #print(CAC_out.shape)
        CAC_out = e4_up + CAC_out    #[256, 16, 16]
        cet_out = self.CAC_conv4(CAC_out)   #[256, 16, 16]
        r1_cat = torch.cat([e3, cet_out], dim=1)   #[512, 16, 16]
        up_out = self.rc_up1(r1_cat)      #[256, 32, 32]
        up5 = self.up5(up_out)      #[128, 32, 32]
        #print('3:',up5.shape)
        r2_cat = torch.cat([e2, up5], dim=1)
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        up_out3 = self.rc_up4(r4_cat)
        up8 = self.up8(up_out3)
        M_Net_out = self.side_8(up8)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        d4 = self.decoder4(e4) + out4 # [256,32,32]
        d3 = self.decoder3(d4) + out3
        d2 = self.decoder2(d3) + out2 # [64,128,128]
        d11 = self.decoder1(d2)
        d1 = self.decoder1(d2) + out1 # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        ave_out = (cet_out + M_Net_out) /2

        #return F.sigmoid(cet_out),F.sigmoid(M_Net_out),F.sigmoid(ave_out),cet_out,M_Net_out,ave_out
        return F.sigmoid(ave_out)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
        
          
            
class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

class DACblock137(nn.Module):
    def __init__(self, channel):
        super(DACblock137, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=False, BatchNorm=False, is_relu=True, num_groups=32):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        if BatchNorm:
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn:
            if out_channels//num_groups==0:
                num_groups=1
            self.gn  =nn.GroupNorm(num_groups, out_channels, eps=BN_EPS)
        self.is_bn = is_bn
        self.is_BatchNorm=BatchNorm
        if is_relu is False: self.relu=None

    def forward(self,x):
        x = self.conv(x)
        if self.is_BatchNorm: x = self.bn(x)
        if self.is_bn: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x

class VSSblock(nn.Module):
    def __init__(self, dim, pooling=True):
        super(VSSblock, self).__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=0.2,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0.,
                d_state=16,
            )
            for i in range(2)])
        self.pooling = pooling
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.pooling:
            pool = F.max_pool2d(x, kernel_size=2, stride=2)
            return x,pool
        else:
            return x

class M_Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=False, bn=False, BatchNorm=False, num_groups=32):
        super(M_Encoder, self).__init__()
        padding =(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )
        self.pooling = pooling

    def forward(self, x):
        conv = self.encode(x)
        if self.pooling:
            pool = F.max_pool2d(conv, kernel_size=2, stride=2)
            return conv,pool
        else:
            return conv


class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, BatchNorm=False, num_groups=32):
        super(M_Conv, self).__init__()
        padding =(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv = self.encode(x)
        return conv


class M_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv = False, bn=False, BatchNorm=False, num_groups=32):
        super(M_Decoder, self).__init__()
        padding =(dilation*kernel_size-1)//2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                             stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1,is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        out = F.upsample(x, size=(H,W),mode='bilinear')
        out = torch.cat([x_big,out], dim=1)
        if self.deconv:
            out = self.deconv(out)
        else:
            out = self.decode(out)
        return out


class M_Decoder_my_10(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv = False, bn=False, BatchNorm=False, num_groups=32):
        super(M_Decoder_my_10, self).__init__()
        padding =(dilation*kernel_size-1)//2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                             stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1,is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self,x):
        x = self.decode(x)
        return x


if __name__ == '__main__':
    from torchstat import stat

    # model = H_Net(3,2,bn=True, BatchNorm=False)
    model = create_model(arch='DeepLabV3Plus', encoder_name="efficientnet-b0", encoder_weights= "imagenet",
                          in_channels = 3, classes = 2)

    eff = smp.encoders.get_encoder(name='efficientnet-b0', in_channels=3, depth=4, weights=None)
    a = torch.rand((2, 3, 512, 512))
    print(model.encoder._blocks)