import torch
import torch.nn as nn
from hyperparameter import *
import numpy as np
import torch.nn.functional as F

cfg_seperated = [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class seperate_VGG16(nn.Module):
    def __init__(self):
        super(seperate_VGG16, self).__init__()
        self.CM_1 = nn.Sequential(*[nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2)])
        self.CM_2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2)])
        self.CM_3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2)])
        self.CM_4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2)])
        self.CM_5 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2)])
        self.classifier = nn.Linear(512,10)


    def forward(self, x):
        out = F.pad(x,(1,0,1,0))
        out = self.CM_1(out)
        out = self.CM_2(out)
        out = self.CM_3(out)
        out = self.CM_4(out)
        out = self.CM_5(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class LC(nn.Module):
    def __init__(self):
        super(LC,self).__init__()
        self.C = nn.Sequential(*[nn.Conv2d(3, 64, kernel_size=3, padding=1)])
        self.L = nn.Linear(3072, 3072, bias=False)

    def forward(self, x):
        out = x.view(Batch_Size, 1, 3072)
        out = self.L(out)
        out = out.view(Batch_Size, 3, 32, 32)
        out = self.C(out)
        return out

class LCVGG16(nn.Module):
    def __init__(self):
        super(LCVGG16, self).__init__()
        self.comb = np.load("Combination.npy").astype(np.float32)
        self.LC = torch.from_numpy(self.comb).cuda()
        self.rest = rest()
        #for param in self.LC.parameters():
            #param.requires_grad = False

    def forward(self, x):
        out = x.view(Batch_Size, 3072)
        out = out.mm(self.LC)
        out = out.view(Batch_Size, 64, 32, 32)
        out = self.rest(out)
        return out


class rest(nn.Module):
    def __init__(self):
        super(rest, self).__init__()
        self.restconv = self._make_layers()
        if Dataset == "CIFAR10":
            self.classifier = nn.Linear(512, 10)
        else:
            self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.restconv(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 64
        for x in cfg_seperated:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


