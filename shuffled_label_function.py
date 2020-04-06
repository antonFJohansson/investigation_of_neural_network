# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:47:33 2020

@author: johaant
"""

import torch
from torch.utils.data import Dataset
import random

class random_dataset(Dataset):
  
  def __init__(self, train_loader, prop_corr = 0.1, datapoint_ind_store = None):
    super().__init__()
    self.train_loader = train_loader
    self.new_ids = [kkk for kkk in range(len(train_loader.dataset))]
    #self.new_labels = [random.randint(0,9) for kkk in range(len(train_loader.dataset))]
    #random.shuffle(self.new_ids)
    prop_corr_int = int(prop_corr*len(train_loader.dataset))
    self.datapoint_ind_store = datapoint_ind_store
    #start_corr_ind = prop_corr_int
    self.new_labels = [random.randint(0,9) for kkk in range(0, prop_corr_int)] + [train_loader.dataset[kkk][1] for kkk in range(prop_corr_int,len(train_loader.dataset))]
    
  def get_rank_points(self):

    store_data_points = []
    for jjj in self.datapoint_ind_store:
      data_points = self.train_loader.dataset[jjj][0]
      store_data_points.append(data_points)
    return store_data_points

  def __len__(self):
    return len(self.train_loader.dataset)

  def __getitem__(self, idx):

    ## I should just do this procedure once I think.
    new_label = self.new_labels[idx]
    new_id = self.new_ids[idx]
    new_img = self.train_loader.dataset[new_id][0]
    return (new_img, new_label)

def get_random_loader(train_loader, prop_corr, datapoint_ind_store):

  rand_dataset = random_dataset(train_loader, prop_corr,datapoint_ind_store)
  store_data_points = rand_dataset.get_rank_points()
  rand_loader = torch.utils.data.DataLoader(rand_dataset, batch_size=train_loader.batch_size, shuffle= True)
  return rand_loader, store_data_points
    