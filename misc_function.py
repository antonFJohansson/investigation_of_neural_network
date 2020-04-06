# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:35:59 2020

@author: johaant
"""

## My test file

import os

def create_folder():
    folder_id = 0
    while True:
        save_folder = 'Experiment ' + str(folder_id)
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
            break
        folder_id = folder_id + 1
    return save_folder

def write_info_txt(net_dict, save_folder, extra_info = None):
    
    if not extra_info is None:
        print_str = 'Extra Info: ' + extra_info + '\n'
    else:
        print_str = ''
        
    print_str =  print_str + 'Layer name : Layer widths (The save notation of the pickle file is Dataset_Net_lr_epochs) (layer save notation is [num_neurons_in, num_neurons_out] for each layer)\n'
    for key in net_dict.keys():
        curr_list = net_dict[key]
        print_str = print_str +  key + ' : ' + str(curr_list) + '\n'
        save_path = os.path.join(save_folder, 'Information.txt')
        with open(save_path, 'w') as f: 
            f.write(print_str) 
    

def create_net_dict():
    
    ## Create the differnt networks to try here, store their parameters as lists
    num_layers = 4
    net_width1 = [[28*28, 50]] + num_layers*[[50,50]]
    net_width2 = [[28*28, 100]] + num_layers*[[100,100]]
    net_width3 = [[28*28, 200]] + num_layers*[[200,200]]
    #net_width4 = [[28*28, 35]] + num_layers*[[35,35]]
    #net_width5 = [[28*28, 50]] + num_layers*[[50,50]]
    #net_width6 = [[28*28, 250]] + num_layers*[[250,250]]
    
    
    #net_width2 = [[28*28, 300]] + num_layers*[[300,300]]
    #net_width3 = [[28*28, 800]] + num_layers*[[800,800]]
    #net_width4 = [[28*28, 1000]] + num_layers*[[1000,1000]]
    net_layer_list = [net_width1, net_width2, net_width3]#, net_width4, net_width5, net_width6]
    
    NET_list = ['Net' + str(k + 1) for k in range(len(net_layer_list))]
    NET_dict = dict(zip(NET_list, net_layer_list))
    return NET_dict

import torch
def calc_rank(my_net, num_rank_val_samples, data_point_store, rank_on_train_data):
    
    """
    Calculate the rank on a given set of points
    my_net: The neural network
    num_rank_val_sampels: The number of datapoints to caluclate the rank on
    data_point_store. The datapoints where we calculate the rank.
    rank_on_train_data: If we calculate the rank on trian data or not
    """
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    rank_list = []

    ## Get the rank values before training
    for jjj in range(num_rank_val_samples):
      
        ## If True then we calculate the rank on training data points
      if rank_on_train_data == True:
        data_point1 = data_point_store[jjj][0].view(-1,28*28)
        data_point1 = data_point1.to(device)
      else:
        data_point1 = torch.Tensor(1,28*28).normal_(0,3)
        data_point1 = data_point1.to(device)
      
      my_net(data_point1, get_rank = True)

      net_rank = my_net.rank
      rank_list.append(net_rank)
      my_net.set_initial_rank()

    return rank_list













