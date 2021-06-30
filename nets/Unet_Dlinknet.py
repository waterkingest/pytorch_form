import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from nets.vgg import VGG16
from functools import partial

nonlinearity = partial(F.relu,inplace=True)

class Dblock_more_dilate(nn.Module):
    def __init__(self,channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out
import torchvision.utils as vutils
# from tensorboardX import SummaryWriter
# # 定义Summary_Writer
# writer = SummaryWriter('./runs') 

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):

        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class MyUnet(nn.Module):
    def addimage(self,images,name):#可视化featuremap
        if self.record:
            x1 = images.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
            print(x1.size())
            img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)  # normalize进行归一化处理
            writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)
            print('finish')
        else:
            pass
    def __init__(self, num_classes=21, in_channels=3, pretrained=False,record=False):
        super(MyUnet, self).__init__()
        self.vgg = VGG16(pretrained=pretrained,in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        self.record=record
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.Dblock=Dblock_more_dilate(512)
        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat1 = self.vgg.features[  :4 ](inputs)
        self.addimage(feat1,'vgg1')
        feat2 = self.vgg.features[4 :9 ](feat1)
        self.addimage(feat2,'vgg2')
        feat3 = self.vgg.features[9 :16](feat2)
        self.addimage(feat3,'vgg3')
        feat4 = self.vgg.features[16:23](feat3)
        self.addimage(feat4,'vgg4')
        feat5 = self.vgg.features[23:-1](feat4)
        self.addimage(feat5,'vgg5')
        feat5=self.Dblock(feat5)
        up4 = self.up_concat4(feat4, feat5)
        self.addimage(up4,'up4')
        up3 = self.up_concat3(feat3, up4)
        self.addimage(up3,'up3')
        up2 = self.up_concat2(feat2, up3)
        self.addimage(up2,'up2')
        up1 = self.up_concat1(feat1, up2)
        self.addimage(up1,'up1')

        final = self.final(up1)
        # writer.close()
        return final

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

