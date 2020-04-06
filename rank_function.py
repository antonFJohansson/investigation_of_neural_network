# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:28:23 2020

@author: johaant
"""

import numpy as np
import torch


def convex_comb_rank(data_point1, data_point2, alpha, network, device, return_rank = True, thres = 1e-8):

    """
    Function to compute the rank along the line between two points.
    data_point1: The first data point
    data_point2: The second data point
    alpha: The values for the line
    network: The net
    return_rank: If True will return the rank of the matrix, otherwise the singluar values
    """


    data_point = alpha*data_point1 + (1-alpha)*data_point2

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
    if return_rank == True:
      computed_rank = np.linalg.matrix_rank(weight_matrix, tol = thres)
      return computed_rank
    else:
      u, s, v = np.linalg.svd(weight_matrix)
      return s