# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:37:57 2020

@author: johaant
"""

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):


  def __init__(self, width_list):
    super().__init__()
    self.input_dim = width_list[0][0]
    self.output_dim = 10
    
    ## I guess this construction can be generlaized to consecutive layers of different sizes and such as well
    self.fc_list = []
    for layer_width in width_list:
      current_layer = nn.Linear(layer_width[0], layer_width[1])
      self.fc_list.append(current_layer)
    self.fc_list = nn.ModuleList(self.fc_list)
    self.fc_end = nn.Linear(width_list[-1][1],10)
    self.rank = self.output_dim

  def set_initial_rank(self):
    self.rank = self.output_dim

  def get_minimum_neurons(self, x):
    ## Gets the rank by taking the minimum of the number of active neurons

    self.rank = min(self.rank, torch.where(x>0)[1].shape[0])


  def forward(self, x, layer_id = None, node_id = None, get_rank = False):
    x = x.view(-1,28*28)
    
    ## If we just want the regular output of the network
    if not get_rank:
      
      if (layer_id == None) and (node_id == None):
        for fc in self.fc_list:
            x = F.relu(fc(x))
        x = self.fc_end(x)

        return x
      else:
        for idx,fc in enumerate(self.fc_list):
          if idx == layer_id:
            x = fc(x)
            output_x = x[:,node_id]
            return output_x
          else:
            x = F.relu(fc(x))
        x = self.fc_end(x)

        return x
    else:
      if (layer_id == None) and (node_id == None):
        for fc in self.fc_list:
            x = F.relu(fc(x))
            self.get_minimum_neurons(x)
        x = self.fc_end(x)

        return x
