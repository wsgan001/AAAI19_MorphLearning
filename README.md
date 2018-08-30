# AAAI19_MorphLearning
This is an anonymous repo for AAAI-2019 submisssion "Morphed Learning: Towards Privacy-Preserving for Deep Learning Based Applications"

Abstract: The concern of potential privacy violation has prevented efficient use of big data for improving deep learning based applications. In this paper, we propose Morphed Learning, a privacy-preserving technique for deep learning based on data morphing that,allows data providers to share their data without leaking sensitive privacy information. 
Morphed Learning is significant since it addresses the drawbacks of existing privacy-preserving deep learning techniques by providing these three features: (1) Strong protection against reverse-engineering on the morphed data; (2) Acceptable computational and data transmission overhead with no correlation to the depth of the neural network; (3) No degradation of the neural network performance.
Theoretical analyses on CIFAR-10 dataset and VGG-16 network show that our method is capable of providing $10^{89}$ morphing possibilities with only 5\% computational overhead and 10\% transmission overhead under limited knowledge attack scenario. 
Further analyses also proved that our method can provide same resilient against full knowledge attack if provided with more resource.

# How to use

## Step 1: Install dependency   
numpy       1.14.5   
Pillow      5.2.0    
scipy       1.1.0    
torch       0.4.0    
torchvision 0.2.1    
tqdm        4.23.4   

## Step 2: Get the matrix for data morphing and the pretrained VGG16 on cifar-10 dataset
Use command: `python cifar10_pretrain.py`

## Step 3: Get the augmented convolutional layer:
Use command: ` python generate_comb.py`

## Step 4: Assess the ablility to restore the accuracy of the augmented convolutional layer:
Use command: `python without_aug_conv.py` to get the test accuracy of the original VGG16 using morphed data as traning and testing dataset.

Use command: `python Aug_conv_test.py` to get the test accuracy of the aug-conv layer+VGG16 using morphed data as traning and testing dataset.

The default dataset is CIFAR10. If you wish to use CIFAR100, please change the parameter `Dataset` in  `hyperparameter.py` to perform the test on CIFAR100.

## Step 5: Assess the security
To assess the security of our method under the inverse traning senario, use command `python LCreverse_dataset.py` to generate the inverse traning dataset

Then use command `LC_train.py` to train the inverse matrix. 10 of the retrieve images and the orignal images will be saved to your work directory,  and the MSEloss between the original image and the retrieved image will be print out.