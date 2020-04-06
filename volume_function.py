# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 07:20:54 2020

@author: johaant
"""
import torch
import numpy as np

def get_weight_matrix(data_point, network):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
  x = data_point
  x.requires_grad = True

  weight_matrix = np.zeros((network.output_dim, network.input_dim))

  for iii in range(network.output_dim):
    
    out = network(x)
    v_tensor = torch.zeros((1,network.output_dim))
    v_tensor = v_tensor.to(device)
    v_tensor[0,iii] = 1
    out.backward(v_tensor)
    jac_vec = x.grad.view(1,network.input_dim).cpu().numpy()
    weight_matrix[iii,:] = jac_vec
    x.grad.zero_()

  x.requires_grad = False
  wm_t = torch.from_numpy(weight_matrix).float()
  wm_t = wm_t.to(device)
  bias = network(x) -  torch.t(torch.mm(wm_t, torch.t(x)))
  return wm_t, bias
#my_net = Network()
#data_point = torch.Tensor(1,5).normal_()
#wt, bt = get_weight_matrix(data_point, my_net)


## And now we want to get the size of a region
## So we take a datapoint, we look along the directions of the basis vector
## We sample along random directions, with distance r from out point


def calc_vol(org_data_point, dist_r, my_net, num_samples = 1000):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
  W_org, b_org = get_weight_matrix(org_data_point, my_net)

  num_points_in_vol = 0
  for iii in range(num_samples):
    dir_vec = torch.Tensor(org_data_point.shape).normal_()
    dir_vec = dir_vec / torch.sum(dir_vec**2)
    dir_vec = dir_vec.to(device)
    new_data_point = org_data_point + dist_r*dir_vec
    W_new, b_new = get_weight_matrix(new_data_point, my_net)

    bias_equal = torch.allclose(b_org, b_new)
    weights_equal = torch.allclose(W_org, W_new)
    if bias_equal and weights_equal:
      num_points_in_vol = num_points_in_vol + 1

  return num_points_in_vol / float(num_samples)






