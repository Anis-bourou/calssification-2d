import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 

import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms



# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2


# data transformation

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=train_transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=test_transform)


# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# choose the training and test datastets

train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=train_transform)

test_data  = datasets.CIFAR10('data', train=False,
                              download=True, transform=test_transform)

# obtain training indices that will be used for validation 
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]




# define the model

vgg16 = models.vgg16(pretrained=True)
for param in vgg16.features.parameters():
    param.requires_grad = False 

n_inputs = vgg16.classifier[6].in_features
last_layer = nn.Linear(n_inputs, 10)
vgg16.classifier[6] = last_layer
vgg16.cuda()

# specify loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)

# training
n_epochs = 100

for epoch in range(1, n_epochs+1):

    train_loss = 0.0
    val_loss   = 0.0
    acc_train  = []
    acc_val    = []
    ##################
    # train the model#
    ##################
    vgg16.train()
    for batch_i, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = vgg16(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _,prediction = torch.max(output,1)
        acc_train.append((prediction == target).float().sum()/batch_size)
    
    #####################
    #validate the model#
    #####################
    vgg16.eval()
    for batch_i , (data, target) in enumerate(valid_loader):

        data, target = data.cuda(), target.cuda()
        output = vgg16(data)
        loss = criterion(output, target)
        val_loss += loss.item()
        _,prediction2 = torch.max(output,1)
        acc_val.append((prediction2 == target).float().sum()/batch_size)
        print("check")
        print()



    print(f'Epoch{epoch}, loss_train:{train_loss/len(train_loader.sampler)}, loss_val:{val_loss/len(valid_loader.sampler)},train_acc:{sum(acc_train)/len(acc_train)}, val_acc:{sum(acc_val)/len(acc_val)}')



