
import torch
import torch.nn as nn


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNet(nn.Module):
    def __init__(self, in_channels,channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.channels=channels
        self.n_classes = n_classes
        self.up1 = Up(channels)
        self.up2 = Up(channels)
        self.up3 = Up(channels)
        self.up4 = Up(channels)
        self.outc = OutConv2d(channels, n_classes)
        
        self.re1 = ReverseEdge(channels)
        self.re2 = ReverseEdge(channels)
        self.re3 = ReverseEdge(channels)
        self.re4 = ReverseEdge(channels)
        self.down = nn.MaxPool2d(2)
        self.dConv1 = DoubleConv(channels,channels)
        self.dConv2 = DoubleConv(channels,channels)
        self.dConv3 = DoubleConv(channels,channels)
        self.dConv4 = DoubleConv(channels,channels)
        self.dConv5 = DoubleConv(channels,channels)
    def forward(self, x):
        enc1 = self.dConv1(x)
        down1 = self.down(enc1)
        enc2 = self.dConv2(down1)
        x1 = self.re1(enc2, enc1)
        down2 = self.down(enc2)           
        enc3 = self.dConv3(down2)
        x2 = self.re2(enc3, enc2)
        down3 = self.down(enc3)
        enc4 = self.dConv4(down3)
        x3 = self.re3(enc4, enc3)
        down4 = self.down(enc4)
        enc5 = self.dConv5(down4)
        x4 = self.re4(enc5, enc4)
        x = self.up1(enc5, x4)

        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feature = self.up4(x, x1)
        x = self.outc(feature)
        return x,feature
        


class UNet_3Plus(nn.Module):
    def __init__(self, in_channels,channels=64,n_classes=2):
        super(UNet_3Plus, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes=n_classes

        self.conv1 = DoubleConv(self.in_channels, self.channels)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = DoubleConv(self.channels, self.channels)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = DoubleConv(self.channels, self.channels)

        self.CatChannels = self.channels
        self.CatBlocks = 3
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h3_Cat_hd3_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)


        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        self.h2_Cat_hd2_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.relu2d_1 = nn.ReLU(inplace=True)


        self.h1_Cat_hd1_conv = nn.Conv2d(self.channels, self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)  # 16
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.outconv1 = nn.Conv2d(self.CatChannels, n_classes, 3, padding=1)

    def forward(self, inputs):
        h1 = self.conv1(inputs)

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  


        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_conv(h3))
        hd3 = self.relu3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3), 1)))

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_conv(h2))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        hd2 = self.relu2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2), 1)))

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_conv(h1))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        x = self.relu1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1), 1)))

        d1 = self.outconv1(x)
        return d1,x
    
class LA_Net(nn.Module):
    def __init__(self, in_channels,out_channels,plane_perceptron_channels,n_classes,block_size):
        super(LA_Net, self).__init__()
        self.in_channels = in_channels
        self.channels=out_channels
        self.n_classes = n_classes
        self.plane_perception_channels=plane_perceptron_channels
        self.input3d = InConv3d(in_channels,out_channels)
        self.ALAP1 = MutiScaleLayerAttention(out_channels, block_size[0])
        self.UP1 = UndirectionalPooling(8)
        self.ALAP2 = MutiScaleLayerAttention(out_channels, int(block_size[0]/8))
        self.UP2 = UndirectionalPooling(5)
        self.ALAP3 = MutiScaleLayerAttention(out_channels, int(block_size[0]/8/5))
        self.UP3 = UndirectionalPooling(4)
        self.input2d = skip(out_channels,block_size[0])
        self.pp= UNet(out_channels,plane_perceptron_channels, n_classes)
        initialize_weights(self)
    def forward(self, x0):
        x0 = self.input3d(x0)
        x = self.UP1(x0)
        x = self.ALAP2(x)
        x = self.UP2(x)
        x = self.ALAP3(x)
        x = self.UP3(x)
        x = self.input2d(x0,x)
        x,feature = self.pp(x)

        x=torch.unsqueeze(x,2)
        return x,feature


class MutiScaleLayerAttention(nn.Module):
    def __init__(self, channels, depth_size):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 1)
        self.convR0 = nn.Conv3d(channels, channels, 1)
        self.convR1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.convR2 = nn.Conv3d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.convR3 = nn.Conv3d(channels, channels, kernel_size=3, padding=3, dilation=3)
        self.planePooling = nn.AdaptiveAvgPool3d((depth_size, 1, 1))
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.gn = nn.GroupNorm(num_groups=32,num_channels=channels) 
    def forward(self, x):
        x1 = self.relu(self.gn(self.convR0(x)))
        x2 = self.relu(self.gn(self.convR1(x)))
        x3 = self.relu(self.gn(self.convR2(x)))
        x4 = self.relu(self.gn(self.convR3(x)))
        x = torch.add(x1,x2)
        x = torch.add(x,x3)
        x = torch.add(x,x4)
        x = self.relu(self.gn(self.conv1(x)))
        x0 = torch.sigmoid(self.relu(self.gn(self.conv2(self.planePooling(x)))))
        x = torch.add(x,torch.mul(x, x0))
        return x

class UndirectionalPooling(nn.Module):
    def __init__(self, pooling_size):
        super().__init__()
        self.undirectionPooling = nn.MaxPool3d(kernel_size=[pooling_size, 1, 1])
    def forward(self, x):
        return self.undirectionPooling(x)
    
class ReverseEdge(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv1x1 = nn.Conv2d(channels, 1, 1)
        self.up = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1_2 = nn.Conv2d(1, 1, 1)
    def forward(self, x, x2):
        x = self.up(self.conv1x1_2(self.relu(self.conv1x1(x))))
        x = -1 * (torch.sigmoid(x)) + 1
        if x.shape[2] != x2.shape[2]:
            x = self.up2(x)
        x = x.expand(-1, self.channels, -1, -1).mul(x2)
        x = x + x2
        return x
    
    
class DetailFusion(nn.Module):
    def __init__(self, in_channels, pooling_size):
        super().__init__()
        self.avgPooling = nn.AvgPool3d((pooling_size,1,1))
        self.maxPooling = nn.MaxPool3d((pooling_size,1,1))
        self.conv1 = nn.Conv3d(in_channels, 1, 1)

    def forward(self, x, x1):
        x0 = self.conv1(x)
        ave = self.avgPooling(x)
        max = self.maxPooling(x)
        x = torch.cat([ave,max])
        x0 = 1-torch.sigmoid(x0)
        x0 = torch.mul(x, x0)
        x = torch.add(x, x0)
        x = torch.cat(x, x1)
        return x
    

class InConv3d(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32,num_channels=channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.in_conv(x)


class skip(nn.Module):
    def __init__(self, channels,cube_height):
        super().__init__()
        self.double2dconv = DoubleConv(channels*2,channels)
        self.skip_conv  = nn.Conv3d(channels, channels, kernel_size=[cube_height, 1, 1], stride=[cube_height, 1, 1])
    def forward(self, x0, x):
        x1 = self.skip_conv(x0)
        x = torch.cat([x1, x], dim=1)
        x = torch.squeeze(x, 2)
        return self.double2dconv(x)
    

class OutConv2d(nn.Module):
    def __init__(self, channels, n_class):
        super(OutConv2d, self).__init__()
        self.conv = nn.Conv2d(channels, n_class, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    

class OutConv3d(nn.Module):
    def __init__(self, channels,n_class):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv3d(channels, n_class, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.out_conv(x)
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=32,num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=32,num_channels=out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out += residual
        return out
    

class Double3DConv(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.double_3dconv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32,num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32,num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_3dconv(x)    

class Down(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(channels,channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    def __init__(self, channels,  bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(channels // 2, channels // 2, kernel_size=2, stride=2)  
        self.conv = DoubleConv(channels*2,channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
