import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from vgg import LCVGG16
from utils import *
from hyperparameter import *

sys.path.append(os.getcwd())


if __name__ == '__main__':

    # load data
    trainloader, valloader = prepare_morphed_data()

    net = LCVGG16()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device used: ', device)
    net.to(device)
    print('start to train.')
    lr = LR
    for epoch in range(N_Epoch):
        if epoch%20 == 19:
            lr *= 0.5
            print('start to val.')
            val(device, net, valloader)
            print('save model')
            torch.save(net, 'Aug_conv_VGG16.pkl')
        optimizer = optim.SGD(params=net.rest.parameters(), lr=lr, momentum=0.9, weight_decay=2e-3)
        train(net, device, nn.CrossEntropyLoss(), trainloader, optimizer, 1, epoch_count=epoch+1)

    print('start to val.')
    val(device, net, valloader)

    torch.save(net, 'Aug_conv_VGG16.pkl')
    print('save model')
