import torch
import torch.nn as nn
import numpy as np
from encrypt import encrypt_image
from torchvision import transforms
import torchvision


class LC_dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.mulmat = torch.from_numpy(np.load("mul_matrix.npy").astype(np.float32))
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.train_data = np.load("LCreverse_samples.npy")
        self.train_gt = np.load("LCreverse_gt.npy")

    def __getitem__(self, index):
        data = self.train_data[index]
        data = torch.from_numpy(data)
        for i in range(3):
            data[i] = data[i].mm(self.mulmat)
        gt = self.train_gt[index]
        return data, gt

    def __len__(self):
        return 50000


if __name__ == '__main__':

    net = torch.load('pretrained_VGG16.pkl')

    CC = net.CM_1[0]

    function = lambda x:encrypt_image(x)

    transform_train = transforms.Compose([
          transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #transforms.Lambda(function)
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=False, num_workers=2)

    sample = torch.Tensor(50000, 3, 32, 32)
    gt = torch.Tensor(50000, 64, 32, 32)
    print("generating data")

    with torch.no_grad():
        for i, (data, label) in enumerate(trainloader):
            data = data.cuda()
            sample[i*50:(i+1)*50] = data
            gt[i*50:(i+1)*50] = CC(data)

    gt_np = gt.numpy()
    sample_np = sample.numpy()

    np.save("LCreverse_samples.npy", sample_np)
    np.save("LCreverse_gt.npy", gt_np)

    print("finish generating data")