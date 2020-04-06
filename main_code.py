
import os
from load_data_function import load_dataset
from network_function import Network
#from dead_node_function import check_zero_nodes, register_node_hooks
from train_network_function import train_network
import pickle
from misc_function import create_folder, write_info_txt, create_net_dict, calc_rank
#from opt_dead_nodes_function import opt_dead_nodes
#from rank_function import convex_comb_rank
import torch
#from volume_function import calc_vol
from shuffled_label_function import get_random_loader
#import random

save_folder_created = False

NET_dict = create_net_dict() ## The neural network parameters are created here
current_net_gen = NET_dict.keys()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

### The main parameters are here
use_random_labels = True ## Randomize a proportion of the training data
## The corropution rate if we use random labels
if use_random_labels:
    corr_rate_list = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
else:
    corr_rate_list = [0]
rank_on_train_data= True ## Calculate the rank of the training data or on random samples?
num_rank_val_samples = 100 ## The number of datapoints to calculate the rank on
num_epochs = 2 ## The number of epochs to train the network for
lr_list = [0.1, 0.01] ## Learning rates to try
dataset_list = ['MNIST'] ## The datasets to train on ('KMNIST', 'MNIST', 'FashionMNIST') exists
per_batch = True ## If we want the training accuracy per batch
## The string that gets printed in the information file in each Experiment folder
extra_info_string = 'Calculate rank now of experiments use_random_labels = True rank_on_train_data= True. We also calculate the rank on the shuffled points now'

## For random labels it is easiest if datapoint_ind_store is a list [0,1,2,....,] 
## But otherwise it does not matter
datapoint_ind_store = [] ## The indices of the data points where to calculate the rank
for kkk in range(num_rank_val_samples):
  datapoint_ind_store.append(kkk)

## Perform the experiment for the desired parameters
for current_net in current_net_gen:
  for current_dataset in dataset_list:
    for lr in lr_list:
      for corr_rate in corr_rate_list:   
        train_loader, test_loader, train_loader_non_random = load_dataset(current_dataset)
        width_list = NET_dict[current_net]

        train_loader, data_point_store = get_random_loader(train_loader, corr_rate, datapoint_ind_store)
        
        ## Create the network
        my_net = Network(width_list)
        my_net = my_net.to(device)

        store_rank_val = {}
        store_rank_val[len(store_rank_val)] = calc_rank(my_net, 
                       num_rank_val_samples, data_point_store,
                       rank_on_train_data)
        

        my_net, train_info_dict = train_network(my_net, train_loader,
                                                test_loader, store_rank_val,
                                                per_batch, lr = lr,
                                                max_epochs = num_epochs, num_rank_val_samples = num_rank_val_samples,
                                                rank_on_train_data = rank_on_train_data,
                                                data_point_store = data_point_store)
        

        store_rank_val[len(store_rank_val)] = calc_rank(my_net, 
                       num_rank_val_samples, data_point_store, 
                       rank_on_train_data)
        

        ## We store all the information in this dictionary and later we pickle it for storage
        all_info_dict = {'Dataset': current_dataset, 'Current Net': current_net,
                        'Network width': NET_dict[current_net], 'lr': lr,
                        'Num epochs': num_epochs, 'Train info': train_info_dict,
                        'per_batch': per_batch, 'rank_val_bt': store_rank_val}#, 'num_dead_nodes_removed_opt': num_dead_nodes_removed_opt}
        

        ## Create the folder
        if not save_folder_created:
            save_folder = create_folder()
            write_info_txt(NET_dict, save_folder, extra_info_string)
            save_folder_created = True
        
        ## We save all info to a pickle
        my_file_save_name = [current_dataset, current_net, str(lr), str(num_epochs), str(corr_rate)]
        my_file_save_name = '_'.join(my_file_save_name)
        my_file_save_name = my_file_save_name + '.pickle'
        save_PATH = os.path.join(save_folder,my_file_save_name)
        
        with open(save_PATH, 'wb') as handle:
            pickle.dump(all_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        

