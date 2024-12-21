import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt


def nn_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, bias=False)
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # 对图像进行卷积操作
    edge_detect = conv_op(Variable(im))
    # 将输出转换为图片格式
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect


def functional_conv2d(im):
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    edge_detect = F.conv2d(Variable(im), weight.type(torch.cuda.FloatTensor), padding=1)
    # edge_detect = F.conv2d(Variable(im), weight.type(torch.FloatTensor), padding=1)
    edge_detect = torch.from_numpy(edge_detect.detach().cpu().numpy())
    return edge_detect


if __name__ == '__main__':
    from thop import profile
    # m = functional_conv2d()
    input1 = torch.randn(32,1,80,80)
    output1 = functional_conv2d(input1)
    print(output1.shape)
