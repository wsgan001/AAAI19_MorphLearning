import torch
import numpy as np
import os

exist = os.path.exists("mul_matrix.npy")

if exist == False:
    mul_matrix_np = np.random.rand(32, 32)
    reverse_np = np.linalg.inv(mul_matrix_np)
    np.save("mul_matrix.npy", mul_matrix_np)
    np.save("reverse_matrix.npy", reverse_np)

mul_matrix_np = np.load("mul_matrix.npy")
mul_matrix = torch.from_numpy(mul_matrix_np).float()

def encrypt_image(image_tensor):
    viewed_tensor = image_tensor
    result_tensor = torch.randn(3, 32, 32)
    for i in range(3):
        result_tensor[i] = viewed_tensor[i].mm(mul_matrix)
    return result_tensor
