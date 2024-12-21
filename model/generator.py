import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from model.components.norm import BatchInstanceNorm2d as BIN
from model.components.feature_extrator1 import Feature_extractor


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
            out = self.sig(out)
            out = torch.mul(out, out1)
            out = out + output[(i)]
            output.append(out)
        return x + output[(size-1)]


class Rmse1(nn.Module):
    def __init__(self,depth=2,mid_channels=64):
        super(Rmse1,self).__init__()
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


class InceptionDWAttentionModule(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, out_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=1/8, bias=True):
        super().__init__()
        bias = to_2tuple(bias)
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        self.res = Rmse(depth=2, mid_channels=in_channels)
        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias[0])
        # self.norm = IBN(in_channels)
        # self.gelu = nn.GELU()
        self.prelu = nn.PReLU()

    # def forward(self, x):
    #     x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
    #     output1 = torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)
    #     # output = self.norm(output)
    #     output2 = self.prelu(output1) + x  # output = self.gelu(output) * x
    #     output2_id, output2_hw, output2_w, output2_h = torch.split(output2, self.split_indexes, dim=1)
    #     output3 = torch.cat((output2_id, self.dwconv_hw(output2_hw), self.dwconv_w(output2_w), self.dwconv_h(output2_h)), dim=1)
    #     output4 = self.prelu(output3) + output2
    #     # output = self.res(output)
    #     output5 = self.fc(output4)
    #     return output5

    # def forward(self, x):
    #     x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
    #     output1 = torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)
    #     # output = self.norm(output)
    #     output2 = self.prelu(output1) + x  # output = self.gelu(output) * x
    #     # output2_id, output2_hw, output2_w, output2_h = torch.split(output2, self.split_indexes, dim=1)
    #     # output3 = torch.cat((output2_id, self.dwconv_hw(output2_hw), self.dwconv_w(output2_w), self.dwconv_h(output2_h)), dim=1)
    #     # output4 = self.prelu(output3) + output2
    #     # # output = self.res(output)
    #     output5 = self.fc(output2)
    #     return output5


    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        output1 = torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)
        # output = self.norm(output)
        output2 = self.prelu(output1) + x  # output = self.gelu(output) * x

        output2_id, output2_hw, output2_w, output2_h = torch.split(output2, self.split_indexes, dim=1)
        output3 = torch.cat((output2_id, self.dwconv_hw(output2_hw), self.dwconv_w(output2_w), self.dwconv_h(output2_h)), dim=1)
        output4 = self.prelu(output3) + output2
        # # output = self.res(output)

        output3_id, output3_hw, output3_w, output3_h = torch.split(output4, self.split_indexes, dim=1)
        output5 = torch.cat((output3_id, self.dwconv_hw(output3_hw), self.dwconv_w(output3_w), self.dwconv_h(output3_h)), dim=1)
        output6 = self.prelu(output5) + output2

        output7 = self.fc(output6)

        return output7

class SFFM_module(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        self.channel_num = channel_num
        self.linear = nn.Linear(channel_num, channel_num, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_list):
        middle_mask = []
        layer_num = len(input_list)
        for i in range(layer_num):
            middle_temp = torch.mean(input_list[i], dim=[2, 3])
            # print(middle_temp.shape)
            middle_temp = self.linear(middle_temp).reshape([-1, self.channel_num, 1, 1])
            # print(middle_temp.shape)
            middle_mask.append(middle_temp)
        channel_con = torch.cat(middle_mask, dim=2)
        channel_list = list(torch.split(channel_con, 1, 1))
        # print(len(channel_list))
        for i in range(self.channel_num):
            channel_fla = nn.Flatten()(channel_list[i])
            # print(channel_fla.shape)
            channel_sof = self.softmax(channel_fla)
            channel_list[i] = channel_sof.reshape([-1, 1, layer_num, 1])
        channel_con = torch.cat(channel_list, 1)
        attention_list = torch.split(channel_con, 1, 2)
        for i in range(layer_num):
            # print('input_list[i] =',input_list[i].shape)
            # print('attention_list[i] =',attention_list[i].shape)
            input_list[i] = input_list[i] * attention_list[i]
        return input_list


class Generator(nn.Module):
    def __init__(self, in_ch=1, intermediate_ch=64):
        super(Generator, self).__init__()
        self.feature1 = Feature_extractor(in_channels=in_ch, out_channels=intermediate_ch)
        # self.feature2 = Feature_extractor(in_channels=in_ch, out_channels=intermediate_ch)
        self.attention_module1 = InceptionDWAttentionModule(in_channels=intermediate_ch, out_channels=intermediate_ch)
        self.attention_module2 = InceptionDWAttentionModule(in_channels=intermediate_ch * 2, out_channels=intermediate_ch)
        # self.attention_module3 = InceptionDWAttentionModule(in_channels=intermediate_ch * 2, out_channels=intermediate_ch)
        # self.attention_module4 = InceptionDWAttentionModule(in_channels=intermediate_ch * 2, out_channels=intermediate_ch)
        self.sffm1 = SFFM_module(channel_num=intermediate_ch)
        self.sffm2 = SFFM_module(channel_num=intermediate_ch)

        self.rmse1 = Rmse1(depth=1, mid_channels=64)
        self.conv = nn.Conv2d(intermediate_ch, in_ch, kernel_size=1)

    def forward(self, x1, x2):
        sum1 = 0
        feature_list1 = []
        feature1 = self.feature1(x1)
        attention1_1 = self.attention_module1(feature1)
        feature_list1.append(attention1_1)

        attentioncat1_1 = torch.cat((attention1_1, feature1), dim=1)
        attention1_2 = self.attention_module2(attentioncat1_1)
        feature_list1.append(attention1_2)

        attentioncat1_2 = torch.cat((attention1_2, attention1_1), dim=1)
        attention1_3 = self.attention_module2(attentioncat1_2)
        feature_list1.append(attention1_3)

        attentioncat1_3 = torch.cat((attention1_3, attention1_2), dim=1)
        attention1_4 = self.attention_module2(attentioncat1_3)
        feature_list1.append(attention1_4)

        sffm_out1 = self.sffm1(feature_list1)
        for i in range(len(sffm_out1)):
            sum1 += sffm_out1[i]
        out1 = self.conv(self.rmse1(sum1))

        sum2 = 0
        feature_list2 = []
        feature2 = self.feature1(x2)
        attention2_1 = self.attention_module1(feature2)
        feature_list2.append(attention2_1)

        attentioncat2_1 = torch.cat((attention2_1, feature2), dim=1)
        attention2_2 = self.attention_module2(attentioncat2_1)
        feature_list2.append(attention2_2)

        attentioncat2_2 = torch.cat((attention2_2, attention2_1), dim=1)
        attention2_3 = self.attention_module2(attentioncat2_2)
        feature_list2.append(attention2_3)

        attentioncat2_3 = torch.cat((attention2_3, attention2_2), dim=1)
        attention2_4 = self.attention_module2(attentioncat2_3)
        attention2_4_f = attention2_4
        feature_list2.append(attention2_4_f)

        sffm_out2 = self.sffm2(feature_list2)
        for i in range(len(sffm_out2)):
            sum2 += sffm_out2[i]

        out2 = self.conv(self.rmse1(sum2))

        return out1, out2







if __name__ == '__main__':
    from thop import profile
    m = Generator()
    input1 = torch.randn(32,1,80,80)
    input2 = torch.randn(32,1,80,80)
    output1, output2 = m(input1, input2)
    print(output1.shape)
    print(output2.shape)
    flops, params = profile(m, inputs=(input1, input2))
    print(flops/1e9,params/1e6) #flops单位G，para单位M
    print('flops: ', flops, 'params: ', params)
    print('flops: %.6f G, params: %.6f M' % (flops / 1e9, params / 1e6))
