import torch
import numpy as np
import time

net = torch.load("pretrained_VGG16.pkl")
C = net.CM_1
C_1 = C[0]
related_pixel = [((0,0),-33), ((0,1),-32), ((0,2),-31), ((1,0),-1), ((1,1),0), ((1,2),1), ((2,0),31), ((2,1),32), ((2,2),33)]

reverse_matrix = np.load("reverse_matrix.npy").astype(np.float32)
weight_1 = C_1.weight.detach().cpu().numpy()
bias_1 = C_1.bias.detach().cpu().numpy()


Inverse = np.zeros((3072, 3072), dtype=np.float32)
Comb_1 = np.zeros((3072, 64*32*32), dtype=np.float32)


start_time = time.time()

for i in range(96):
    for j in range(32):
        Inverse[i*32+j][i*32:i*32+32] = reverse_matrix[j]

print("Inverse matrix complete! Time:", time.time()-start_time)


for m in range(0, 64):
    for n in range(0, 1024):
        line_n = n%32
        for idx, l in related_pixel:
            if (n+l) >= 0 and (n+l) < 1024 and (line_n+idx[1]>0) and (line_n+idx[1]<33):
                Comb_1[n+l][m*1024+n] = weight_1[m][0][idx]
                Comb_1[n+l+1024][m*1024+n] = weight_1[m][1][idx]
                Comb_1[n+l+2048][m*1024+n] = weight_1[m][2][idx]

print("Conv_1 complete! Time:", time.time()-start_time)


Inverse = torch.from_numpy(Inverse)
Inverse = Inverse.cuda()

Comb_1 = torch.from_numpy(Comb_1)
Comb_1 = Comb_1.cuda()

Comb_1 = torch.mm(Inverse, Comb_1)

print("Comb_1 complete! Time:", time.time()-start_time)

Comb_1 = Comb_1.cpu().numpy()
Comb_1 = Comb_1.astype(np.float16)
np.save("Combination.npy", Comb_1)
np.save("Bias.npy", bias_1)

print("Comb_1 Saved:", time.time()-start_time)
