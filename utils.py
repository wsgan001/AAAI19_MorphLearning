import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
from hyperparameter import *
from encrypt import encrypt_image
import numpy as np

use_cuda = torch.cuda.is_available()

function = lambda x:encrypt_image(x)

def val(device, net, valloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predict = outputs.max(1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    # print the accuracy
    print('Accuracy of the network on the val images: %.3f %%' % (
        100 * correct / total))


def train(model, device, criterion, trainloader, optimizer, epochs, epoch_count=None):
    model.train()
    for epoch in range(epochs):
        epoch += 1
        pbar = tqdm(trainloader, total=len(trainloader))
        train_loss_all = .0

        epoch_print = epoch if epoch_count is None else epoch_count
        for batch_id, (inputs, labels) in enumerate(pbar):

            if use_cuda:
                inputs = inputs.to(device)
                labels = labels.to(device)

            inputs = torch.autograd.Variable(inputs)
            labels = torch.autograd.Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predict = outputs.max(1)
            train_loss_all += loss.data
            train_loss = train_loss_all/(batch_id+1)
            pbar.set_description("poch: {%d} - loss: {%5f} " % (epoch_print, train_loss))
    return


def prepare_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Lambda(function)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Lambda(function)
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size, shuffle=True, num_workers=2)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    valloader = torch.utils.data.DataLoader(valset, batch_size=Batch_Size, shuffle=False, num_workers=2)
    return trainloader, valloader


def prepare_morphed_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(function)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(function)
    ])
    if Dataset == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=Batch_Size, shuffle=False, num_workers=2)
    return trainloader, valloader

def datareverse(data):
    trans_mat = np.load("mul_matrix.npy")
    reverse_m_np = np.linalg.inv(trans_mat)
    reverse_m = torch.from_numpy(reverse_m_np).float()
    for i in range(3):
        data[i] = data[i].mm(reverse_m)
    return data


