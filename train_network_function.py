# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:15:50 2020

@author: johaant
"""

import torch
import torch.nn as nn
from misc_function import calc_rank

def train_network(my_net, train_loader, test_loader, store_rank_val, per_batch = False,lr = 1e-3, max_epochs = 15, num_rank_val_samples = 100, rank_on_train_data = True, datapoint_ind_store = None, data_point_store = None):
        
    """
    Function to train the network:
        my_net: The neural network
        train_loader: The dataloader for the training data
        test_loader: The dataloader for the validation data
        store_rank_val_bt: The dictionary for storing the calculated rank values
        per_batch: If we want to store the accuracy per batch or per epoch
        lr: the learning rate
        max_epoch: Maximum num of epochs
        num_rank_val_samples: The number of data points to calcualte the rank
        rank_on_train_data: If we want to calculate the rank on train data
        datapoint_ind_store: The indicies for the datapoints (might not be needed)
        data_point_store: The stored datapoints if we want to calculate the rank on the train data
        
    """
    
    
    
    ## Train the network should also be its own function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    opt = torch.optim.SGD(my_net.parameters(), lr = lr, momentum = 0.8)
    loss_fn = nn.CrossEntropyLoss()
    train_info_dict = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        
    for epoch in range(max_epochs):
                
          for idx, (data, label) in enumerate(train_loader):
            
            my_net.zero_grad()
            
            x = data
            y = label
            
            x = x.to(device)
            y = y.to(device)

            
            out = my_net(x)
            loss = loss_fn(out,y)
            loss.backward()
            opt.step()

            if per_batch == True:
              train_info_dict['train_loss'].append(loss.item())
              
          ## Calculate test acc
          with torch.no_grad():
            acc = 0.
            for data, label in test_loader:
              x = data
              y = label

              x = x.to(device)
              y = y.to(device)

              out = my_net(x)
              loss_v = loss_fn(out,y)
              _, ind = torch.max(out, 1)              
              
              acc = acc + torch.sum(y == ind).float()
              
            final_acc_val = acc/len(test_loader.dataset)
            ## Calculate train acc
            acc = 0.
            for data, label in train_loader:
              x = data
              y = label

              x = x.to(device)
              y = y.to(device)

              out = my_net(x)
              loss_t = loss_fn(out,y)
              _, ind = torch.max(out, 1)
        
              
              acc = acc + torch.sum(y == ind).float()

            final_acc = acc/len(train_loader.dataset)

          store_rank_val[len(store_rank_val)] = calc_rank(my_net, 
                       num_rank_val_samples, data_point_store, 
                       rank_on_train_data)

          #if epoch % (max_epochs // 10) == 0 or epoch == (max_epochs - 1):
          print('Epoch {} Val Acc {} Train Acc {} Train Loss {}'.format(epoch, final_acc_val, final_acc, loss.item()))
          
          train_info_dict['train_acc'].append(final_acc.item())
          train_info_dict['val_acc'].append(final_acc_val.item())
          ##
          train_info_dict['val_loss'].append(loss_v.item())
          if per_batch==False:
            
            train_info_dict['train_loss'].append(loss_t.item())

    return my_net, train_info_dict