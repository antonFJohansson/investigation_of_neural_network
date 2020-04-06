# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:45:14 2020

@author: johaant
"""

import torch
import copy
from random import randint, random

class check_zero_nodes():
                
    """
    Class to find dead nodes in the networks
    """
              
    def __init__(self, my_net, train_loader, test_loader):
      self.dead_nodes_index = []
      self.init = True
      self.curr_layer = 0
      self.train_loader = train_loader
      self.test_loader = test_loader
      ## We will save the history as [Epoch, [[dead nodes layer 1], [dead nodes layer 2], [... etc]]]
      self.dead_node_history = []
      self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

      ## Set all neuron indices to be dead at first
      ## During training we will remove them one by one when they are not dead
      for module in my_net.fc_list:
        num_output_nodes = module.weight.shape[0]
        self.dead_nodes_index.append([kkk for kkk in range(num_output_nodes)])
    
    def rem_index(self, pos_nodes_list):
      for pos_ind in pos_nodes_list:
        try:
          self.dead_nodes_index[self.curr_layer].remove(pos_ind)
        except:
          pass
      self.curr_layer +=1
      if self.curr_layer > (len(self.dead_nodes_index) - 1):
        self.curr_layer = 0
        
    
    ## I should have a way to store the neurons as well when I call this function so I can display how the neurons die during and after training and such as wlel.
    def display_dead_nodes(self, my_net):
        
        ## Have to pass through all data to see which nodes might be dead
        for data, _ in self.train_loader:
                
            x = data
            x = x.to(self.device)
            out = my_net(x)
            
        for data, _ in self.test_loader:
        
            x = data
            x = x.to(self.device)
            out = my_net(x)
                
        for idx,li in enumerate(self.dead_nodes_index):
            str1 = 'Number of potentially dead neurons in layer {} is {}'.format(idx+1, len(li))
            str2 = 'These are {}'.format(li)
          
            print(str1)
            print(str2)
            
    ## -2 for th epoch if we do not input any epoch or anything
    ## We save the information about which nodes that might be dead so we can plot the info later
    def save_dead_nodes_info(self, my_net, epoch = -2):
        

        #for data, _ in self.train_loader:
        #    x = data
        #    x = x.to(self.device)
        #    out = my_net(x)
            
        #for data, _ in self.test_loader:
        
        #    x = data
        #    x=x.to(self.device)
        #    out = my_net(x)
                

        save_list = [epoch, copy.deepcopy(self.dead_nodes_index)]
        self.dead_node_history.append(save_list)


    def random_data_check(self, my_net, random_data = False):
      
        initial_list = copy.deepcopy(self.dead_nodes_index)

        if random_data == False:
          for data, _ in self.train_loader:
            x = data
            x = x.to(self.device)
            out = my_net(x)
            
        for data, _ in self.test_loader:
        
            x = data
            x=x.to(self.device)
            out = my_net(x)

        if random_data == True:
          for data, label in self.train_loader:
            ## This can be done more efficiently but for now it works and creates new random points that we can insert into the network
            for iii in range(data.shape[0]):
              new_tensor = torch.Tensor(data.shape).normal_()
              ## How many datapoints to create a convex combination from
              num_data_points = randint(1,10)
              num_data_point_sample = list(set([randint(0,data.shape[0]-1) for kkk in range(num_data_points)]))
              curr_data_point = data[num_data_point_sample,:,:,:]
              sample_weights = [random() for kkk in range(len(num_data_point_sample))]
              torch_weights = torch.tensor(sample_weights)
              torch_weights = torch.tensor(sample_weights)/torch.sum(torch.tensor(sample_weights))
              curr_data_point = curr_data_point.view(curr_data_point.shape[1],curr_data_point.shape[2],curr_data_point.shape[3],curr_data_point.shape[0])
              new_data_point = curr_data_point*torch_weights
              new_data_point = new_data_point.view(curr_data_point.shape[3],curr_data_point.shape[0],curr_data_point.shape[1],curr_data_point.shape[2])
              new_data_point = torch.sum(new_data_point, dim = 0)
              new_tensor[iii,:,:,:] = new_data_point
            new_tensor=new_tensor.to(self.device)
            out = my_net(new_tensor)
          for data, label in self.test_loader:
            ## This can be done more efficiently but for now it works and creates new random points that we can insert into the network
            for iii in range(data.shape[0]):
              new_tensor = torch.Tensor(data.shape).normal_()
              ## How many datapoints to create a convex combination from
              num_data_points = randint(1,10)
              num_data_point_sample = list(set([randint(0,data.shape[0]-1) for kkk in range(num_data_points)]))
              curr_data_point = data[num_data_point_sample,:,:,:]
              sample_weights = [random() for kkk in range(len(num_data_point_sample))]
              torch_weights = torch.tensor(sample_weights)
              torch_weights = torch.tensor(sample_weights)/torch.sum(torch.tensor(sample_weights))
              curr_data_point = curr_data_point.view(curr_data_point.shape[1],curr_data_point.shape[2],curr_data_point.shape[3],curr_data_point.shape[0])
              new_data_point = curr_data_point*torch_weights
              new_data_point = new_data_point.view(curr_data_point.shape[3],curr_data_point.shape[0],curr_data_point.shape[1],curr_data_point.shape[2])
              new_data_point = torch.sum(new_data_point, dim = 0)
              new_tensor[iii,:,:,:] = new_data_point
            new_tensor=new_tensor.to(self.device)
            out = my_net(new_tensor)

        for idx,li in enumerate(initial_list):
            if not li == self.dead_nodes_index[idx]:
              print('A new dead node was discovered')
              break
        

        
def register_node_hooks(my_net, node_info):
    def get_dead_node_hook(module, input, output, node_info = node_info):
    
        ## Define the hook for the layers that checks for dead nodes
    
        pos_nodes = torch.sum(output<=0, dim = 0)
        pos_nodes = pos_nodes.nonzero() 
        pos_nodes_list = pos_nodes[:,0].tolist() 
        node_info.rem_index(pos_nodes_list)  
    
    for mod in my_net.fc_list:
        mod.register_forward_hook(get_dead_node_hook)   
        
    

    