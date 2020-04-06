# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:33:07 2020

@author: johaant
"""

import torch
import torchvision



def load_dataset(current_dataset = 'MNIST', batch_size = 64):
    
    """
    Code to load a specific dataset
    Choices:
        MNIST
        FashionMNIST
        KMNIST
        
    Returns:
        Data loaders with the data.
        train and test loader are as usual
        train_loader_non_random is just a train loader where the data points are non-random
    
    """
    
    if current_dataset == 'MNIST':
      transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
      train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
      test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformImg)  
    elif current_dataset == 'FashionMNIST':
      transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.286,), (0.3527,))])
      train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transformImg)
      test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transformImg)  
    elif current_dataset == 'KMNIST':
      transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1918,), (0.3482,))])
      train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transformImg)
      test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transformImg)  
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle= True)  
    train_loader_non_random = torch.utils.data.DataLoader(train, batch_size=64)  
    test_loader = torch.utils.data.DataLoader(test)
    
    return train_loader, test_loader, train_loader_non_random