import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from labml_helpers.module import Module
import math
from torch.nn import Conv2d
import numpy as np
from torchvision.models import vgg19


class MixLoss(nn.Module):
    def __init__(self):
        super(MixLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.mseloss = nn.MSELoss()

    def forward(self, pred, label):
        loss = 0.8 * self.l1(pred, label) + 0.2 * self.mseloss(pred, label)
        return loss


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


# img must be variable with grad and of dim N*C*W*H
class TVLossL1(Module):
    def __init__(self):
        super(TVLossL1, self).__init__()
        self.cbloss = L1_Charbonnier_loss()

    def forward(self, img: torch.Tensor):
        img = torch.cat((img, img, img), dim=1)
        hor = grad_conv_hor()(img)
        vet = grad_conv_vet()(img)
        target = torch.autograd.Variable(torch.FloatTensor(img.shape).zero_().cuda())

        loss_hor = F.l1_loss(hor, target)
        loss_vet = F.l1_loss(vet, target)

        loss = (loss_hor+loss_vet)  #  / 3.0    # 3.0
        return loss


# horizontal gradient, the input_channel is default to 3
def grad_conv_hor():
    grad = Conv2d(3, 3, (1, 3), stride=1, padding=(0, 1))
    weight = np.zeros((3, 3, 1, 3))

    for i in range(3):
        weight[i, i, :, :] = np.array([[-1, 1, 0]])

    weight = torch.FloatTensor(weight).cuda()
    weight = nn.Parameter(weight, requires_grad=False)
    bias = np.array([0, 0, 0])
    bias = torch.FloatTensor(bias).cuda()
    bias = nn.Parameter(bias, requires_grad=False)
    grad.weight = weight
    grad.bias = bias
    return grad


# vertical gradient, the input_channel is default to 3
def grad_conv_vet():
    grad = Conv2d(3, 3, (3, 1), stride=1, padding=(1, 0))
    weight = np.zeros((3, 3, 3, 1))

    for i in range(1):
        weight[i, i, :, :] = np.array([[-1, 1, 0]]).T

    weight = torch.FloatTensor(weight).cuda()
    weight = nn.Parameter(weight, requires_grad=False)
    bias = np.array([0, 0, 0])
    bias = torch.FloatTensor(bias).cuda()
    bias = nn.Parameter(bias, requires_grad=False)
    grad.weight = weight
    grad.bias = bias
    return grad


class VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=False)

        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
        out = self.feature_extractor(x)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
    # def __init__(self):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 32):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = L1_Charbonnier_loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x1 = torch.cat((x, x, x), dim=1)
        y1 = torch.cat((y, y, y), dim=1)
        x_vgg, y_vgg = self.vgg(x1), self.vgg(y1)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


