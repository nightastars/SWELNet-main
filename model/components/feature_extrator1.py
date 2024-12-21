import torch
import torch.nn.functional as F
from torch import nn


class Rmse(nn.Module):
    def __init__(self,depth=2,mid_channels=64):
        super(Rmse,self).__init__()
        self.resblock = nn.Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1,stride=1)
        self.sig = nn.Sigmoid()
        self.conv =nn.Sequential(nn.Conv2d(mid_channels,mid_channels,kernel_size=1,padding=0,stride=1),nn.PReLU(mid_channels),
                                 nn.Conv2d(mid_channels,mid_channels,kernel_size=1,padding=0,stride=1))
        self.depth = str(depth)

    def forward(self,x):
        output = []
        output.append(x)
        size = len(self.depth)
        for i in range(size):
            out1 = self.resblock(output[i])
            out = nn.AdaptiveAvgPool2d((1,1))(out1)
            out = self.conv(out)
            # out = self.sig(out)
            # out = torch.mul(out, out1)
            # out = out + output[(i)]
            output.append(out)
        return x + output[(size-1)]


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # self.gelu = nn.GELU()
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        out = self.prelu(x)
        return out


# # 特征提取器
# class Feature_extractor(nn.Module):
#     # referred from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
#     def __init__(self, in_channels, out_channels, pool_features=1):
#         super(Feature_extractor, self).__init__()
#         self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
#         self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
#         self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
#         self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
#         self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
#         self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
#         self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
#         self.conv = BasicConv2d(225, out_channels, kernel_size=1, padding=0)   # 32, 48, 64
#         self.res = Rmse(depth=1, mid_channels=out_channels)
#
#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)
#         branch5x5 = self.branch5x5_1(x)
#         branch5x5 = self.branch5x5_2(branch5x5)
#         branch3x3dbl = self.branch3x3dbl_1(x)
#         branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
#         branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)
#         output = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
#         output = torch.cat(output, 1)
#         output = self.res(self.conv(output))
#         return output


# 特征提取器
class Feature_extractor(nn.Module):
    # referred from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
    def __init__(self, in_channels, out_channels, pool_features=1):
        super(Feature_extractor, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(16, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_2 = BasicConv2d(16, 64, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(16, 16, kernel_size=1)
        self.conv1 = BasicConv2d(192, 64, kernel_size=3, padding=1)   # 32, 48, 64
        self.res1 = Rmse(depth=1, mid_channels=64)
        self.res2 = Rmse(depth=1, mid_channels=64)
        self.conv2 = BasicConv2d(64, out_channels, kernel_size=1, padding=0)  # 32, 48, 64

    def forward(self, x):
        conv0 = self.conv0(x)
        split = torch.split(conv0, 16, dim=1)
        branch3x3_1 = self.branch3x3_1(split[1])
        branch3x3dbl = self.branch3x3dbl_2(split[2])
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(split[3], kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        output = [split[0], branch3x3_1, branch3x3dbl, branch_pool]
        output = torch.cat(output, 1)
        output = self.conv1(output)
        output = self.res2(self.res1(output))
        output = self.conv2(output)
        return output


if __name__ == '__main__':
    # from thop import profile
    m = Feature_extractor(1, 64)
    input1 = torch.randn(2,1,80,80)
    output1 = m(input1)
    print(output1.shape)
