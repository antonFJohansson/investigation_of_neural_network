# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:10:51 2020

@author: Anton
"""

import copy
import torch
from network_function import Network

def opt_dead_nodes(my_net, node_info, width_list, num_cycles = 2, num_iters_opt = 100, opt_lr = 1):

  num_nodes_removed = 0
  copied_model = Network(width_list)
  copied_model.load_state_dict(my_net.state_dict())
  for param in copied_model.parameters():
    param.requires_grad = False

  for cycle in range(num_cycles):
    print('Cycle {} / {}'.format(cycle, num_cycles-1))
    dead_node_list_copy = copy.deepcopy(node_info.dead_nodes_index)
    num_layers = len(dead_node_list_copy)

    for layer_ind in range(num_layers):
      for curr_di in dead_node_list_copy[layer_ind]:
        x_test = torch.Tensor(1,28,28).normal_()
        x_test.requires_grad = True
        opt_x = torch.optim.SGD([x_test], lr = opt_lr) ## But this one tries to minimize it right? ## descent so yea, but why do we get this then?
        ## I print the negative of it right, but I should print just the out
        for kkk in range(num_iters_opt):
          out = -copied_model(x_test, layer_ind, curr_di)
          if (-out.item()) > 0:
            ## Node is not completely dead
            #print('Node {} in layer {} is not dead'.format(curr_di, layer_ind))
            node_info.dead_nodes_index[layer_ind].remove(curr_di)
            num_nodes_removed = num_nodes_removed + 1
            break
          out.backward()
          opt_x.step()
  ## SO we can later also see how many dead neurons were removed by this procesdure and where with the plot as before
  opt_epoch = node_info.dead_node_history[-1][0] + 1
  node_info.dead_node_history.append([opt_epoch, copy.deepcopy(node_info.dead_nodes_index)])
  return num_nodes_removed