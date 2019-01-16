# import torch
from torch import nn
from torchvision import models


class FeatureExtractModels(nn.Module):
    def __init__(self, model):
        super(FeatureExtractModels, self).__init__()
        if model == 'vgg19':
            vgg19 = models.vgg19_bn(pretrained=True)
            self.feature = nn.Sequential(*list(vgg19.children())[:-1])
            self.feature.add_module('global average', nn.AvgPool2d(7))

        elif model == 'inceptionv3':
            inceptionv3 = models.inception_v3(pretrained=True)
            self.feature = nn.Sequential(*list(inceptionv3.children())[:-1])
            self.feature.add_module('global average', nn.AvgPool2d(11))

        elif model == 'resnet50':
            resnet50 = models.resnet50(pretrained=True)
            self.feature = nn.Sequential(*list(resnet50.children())[:-1])

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x
