import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding, groups=1):
        super(LinearBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding, groups, residual=False):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.conv = ConvBlock(in_c, groups, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_dw = ConvBlock(groups, groups, kernel=kernel, stride=stride, padding=padding, groups=groups)
        self.project = LinearBlock(groups, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        out = self.conv(x)
        out = self.conv_dw(out)
        out = self.project(out)
        if self.residual:
            return out + x
        return out

class Residual(nn.Module):
    def __init__(self, c, num_block, groups_list, kernel, stride, padding):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            groups = groups_list[i] if isinstance(groups_list, list) else groups_list
            # BÊN TRONG CỤM RESIDUAL, LUÔN CÓ SHORTCUT (residual=True)
            modules.append(DepthWise(c, c, kernel, stride, padding, groups, residual=True))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        # Không còn cộng ở đây nữa vì đã cộng trong DepthWise
        return self.model(x)

class MiniFASNet(nn.Module):
    def __init__(self, keep_top_k=2, embedding_size=128, conv6_kernel=(5, 5),
                 drop_p=0.5, num_classes=3, img_channel=3):
        super(MiniFASNet, self).__init__()
        
        # Naming and architecture tailored to match the provided .pth model weights
        self.conv1 = ConvBlock(img_channel, 32, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(32, 32, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
        
        # Transition 2 -> 3 (Không có residual vì thay đổi chiều/kích thước)
        self.conv_23 = DepthWise(32, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=103, residual=False)
        # Stage 3
        self.conv_3 = Residual(64, num_block=4, groups_list=[13, 13, 13, 13], 
                              kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Transition 3 -> 4
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=231, residual=False)
        # Stage 4
        self.conv_4 = Residual(128, num_block=6, groups_list=[231, 52, 26, 77, 26, 26], 
                              kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Transition 4 -> 5
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=308, residual=False)
        # Stage 5
        self.conv_5 = Residual(128, num_block=2, groups_list=[26, 26], 
                              kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Head
        self.conv_6_sep = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = LinearBlock(512, 512, kernel=conv6_kernel, stride=(1, 1), padding=(0, 0), groups=512)
        
        self.flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(drop_p)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out

def get_minifasnet_v2(num_classes=3):
    # This specific setup matches the 2.7_80x80_MiniFASNetV2.pth provided
    return MiniFASNet(conv6_kernel=(5, 5), num_classes=num_classes)
