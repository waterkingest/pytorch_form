import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from torchsummary import summary
# from torchsummary import summary
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
# 定义Summary_Writer
writer = SummaryWriter('./runs') 
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_size),
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_size),
            )
    def forward(self, inputs1, inputs2):
        outputs=self.up(inputs2)
        outputs=self.conv1(outputs)
        outputs = torch.cat([inputs1,outputs], 1)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        return outputs
class unetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetDown3, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_size),
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_size),
            )
        self.Drop=nn.Dropout(0.5)
    def forward(self, inputs1):
        outputs =self.layer1(inputs1)
        outputs =self.layer2(outputs) 
        outputs=self.Drop(outputs)
        # outputs = self.conv2(outputs)
        return outputs
class unetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetDown2, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_size),
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_size),
            )
        self.Drop=nn.Dropout(0.5)
        
    def forward(self, inputs1):
        outputs =self.layer1(inputs1)
        outputs =self.layer2(outputs)
        outputs=self.Drop(outputs)
 
        # outputs = self.conv2(outputs)
        return outputs

class unetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetDown, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_size),
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_size),
            )
    def forward(self, inputs1):
        outputs =self.layer1(inputs1)
        outputs =self.layer2(outputs) 
        # outputs = self.conv2(outputs)
        return outputs


class MyUnet2(nn.Module):
    def addimage(self,images,name):#可视化featuremap
        if self.record:
            x1 = images.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
            print(x1.size())
            img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)  # normalize进行归一化处理
            writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)
            print('finish')
        else:
            pass
    def __init__(self, num_classes=2, in_channels=3, pretrained=False,record=False):
        super(MyUnet2, self).__init__()
        # self.vgg = VGG16(pretrained=pretrained,in_channels=in_channels)
        # in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512,1024]
        resnet = models.resnet34(pretrained=True)
        self.record=record
        self.encoder1 = resnet.layer1
        self.encodecov1=nn.Sequential(
                nn.Conv2d(out_filters[0],out_filters[1], kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_filters[1]),
            )
        self.encoder2 = resnet.layer2
        self.encodecov2=nn.Sequential(
                nn.Conv2d(out_filters[1],out_filters[2], kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_filters[2]),
            )
        self.encoder3 = resnet.layer3
        self.encodecov3=nn.Sequential(
                nn.Conv2d(out_filters[2],out_filters[3], kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_filters[3]),
            )
        self.encoder4 = resnet.layer4
        self.encodecov4=nn.Conv2d(out_filters[3],out_filters[4],kernel_size=3,padding=1)
        self.Down1 = unetDown(in_channels, out_filters[0])

        self.Down2 = unetDown(out_filters[0], out_filters[1])

        self.Down3 = unetDown(out_filters[1], out_filters[2])
        
        self.Down4 = unetDown2(out_filters[2], out_filters[3])
        
        self.Down5 = unetDown3(out_filters[3], out_filters[4])

        self.Up1=unetUp(out_filters[4], out_filters[3])

        self.Up2=unetUp(out_filters[3], out_filters[2])

        self.Up3=unetUp(out_filters[2], out_filters[1])

        self.Up4=unetUp(out_filters[1], out_filters[0])
        # final conv (without any concat)
        self.cov111=nn.Conv2d(out_filters[0],2,kernel_size=3,padding=1)
        self.relu=nn.ReLU(True)
        self.final = nn.Conv2d(2, num_classes, 1)
        self.maxp=nn.MaxPool2d(kernel_size=2, stride=2)
        self.finalsof=nn.Softmax(dim=-1)
    def forward(self, inputs):
        feate1=self.Down1(inputs)
        feate11=self.maxp(feate1)
        self.addimage(feate11,'maxp1')
        e1 = self.encoder1(feate11)#64,512,512
        self.addimage(e1,'res1')
        e11=self.encodecov1(e1)
        e2 = self.encoder2(e1)#128,256,256
        self.addimage(e2,'res2')
        e21=self.encodecov2(e2)
        e3 = self.encoder3(e2)#256，128，128
        self.addimage(e3,'res3')
        e31=self.encodecov3(e3)
        e4 = self.encoder4(e3)#512，64，64
        self.addimage(e4,'res4')
        # feate2=self.Down2(feate11)
        # feate21=self.maxp(feate2)

        # feate3=self.Down3(feate21)
        # feate31=self.maxp(feate3)

        # feate4=self.Down4(feate31)
        # feate41=self.maxp(feate4)

        feate5=self.Down5(e4)

        Up6=self.Up1(e31,feate5)
        self.addimage(Up6,'Up1')
        Up7=self.Up2(e21,Up6)
        self.addimage(Up7,'Up2')
        Up8=self.Up3(e11,Up7)
        self.addimage(Up8,'Up3')
        Up9=self.Up4(feate1,Up8)
        self.addimage(Up9,'Up4')
        final=self.cov111(Up9)
        final=self.relu(final)
        final=self.final(final)
        self.addimage(final,'Final')

        # final=self.finalsof(final)
        writer.close()
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