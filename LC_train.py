import torch
from torchvision import transforms
from LCreverse_dataset import LC_dataset
from hyperparameter import *
from utils import *
from vgg import LC
from torch import optim
from torch import nn
from encrypt import encrypt_image
from scipy.misc import imsave

trainset = LC_dataset(root="./", transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = LC()
net.to(device)
lr = LR
for epoch in range(N_Epoch):
    if epoch%30 == 29:
        lr *= 0.5
        torch.save(net, 'retrieve_LC.pkl')
        print("model saved")
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train(net, device, nn.MSELoss(), trainloader, optimizer, 1, epoch_count=epoch + 1)

torch.save(net, 'retrieve_LC.pkl')
print("model saved")


function = lambda x:encrypt_image(x)

transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(function)
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=False, num_workers=2)

name_list = ['1', '2', '3', '4', ' 5', '6', '7', '8', '9', '10']

net = torch.load("retrieve_LC.pkl")
print("Retrieve matrix loaded.")
L = net.L
net.cpu()
MSEloss = torch.nn.MSELoss(size_average=True)

for i, data_batch in enumerate(trainloader):
    if i == 10:
        with torch.no_grad():
            data, lable = data_batch
            data = data.cpu()
            lable = lable.cpu()
            image = data[0:10]
            image = image.view(10, 1, 3072)
            image = L(image)
            image = image.view(10, 3, 32, 32)
            image.cpu()
            for i in range(10):
                retrieve = image[i]
                original = datareverse(data[i])
                loss = MSEloss(retrieve, original)
                print("The MSEloss between the %d th original image and the retrieved image is %.5f " % (i, loss))
                retrieve = retrieve.numpy()
                original = original.numpy()
                retrieve = np.rollaxis(retrieve, 0, 3)
                original = np.rollaxis(original, 0, 3)
                name_retrieve = 'retrieve_'+name_list[i]+'.png'
                name_original = 'original_'+name_list[i]+'.png'
                imsave(name_retrieve, retrieve)
                imsave(name_original, original)
            break