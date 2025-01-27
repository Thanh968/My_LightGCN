import pandas as pd
from torch import nn, optim
from model import PairWiseModel
from data import BasicDataset
import numpy as np
import random
import torch

class BPR_Loss:
    def __init__(self, recmodel : PairWiseModel, config: dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr = self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)

        reg_loss = self.weight_decay * reg_loss
        loss = loss + reg_loss
        
        self.opt.zero_grad()
        loss.backward()
        
        self.opt.step()

        return loss.cpu().item()

def UniformSample_original_python(dataset: BasicDataset):
    num_users = dataset.num_users
    allPos = dataset.allPos
    num_items = dataset.num_items
    all_items_set = set(list(range(num_items)))
    S = []
    
    for user in range(num_users):
        all_pos_items = allPos[user]
        selected_pos = random.choice(all_pos_items)
        all_pos_set = set(all_pos_items)
        neg_items = all_items_set - all_pos_set
        selected_neg = random.choice(list(neg_items))
        S.append([user, selected_pos, selected_neg])

    return np.array(S)

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def shuffle(*array, **kwargs):
    required_index = kwargs.get('indices', False)
    length_set = set(len(x) for x in array)

    print(f"Tap cac kich thuoc cua tensor: {length_set}")

    if (len(list(length_set)) != 1):
        raise ValueError('Cac tensor phai co cung kich thuoc')
    
    indices_array = np.arange(len(array[0]))
    np.random.shuffle(indices_array)

    result = None

    if len(array) == 1:
        result = array[0][indices_array]
    else:
        result = tuple(x[indices_array] for x in array)

    if (required_index):
        return result, indices_array
    
    return result

def minibatch(*tensor, **kwargs):
    batch_size = kwargs.get('batch_size', 94)
    for i in range(0, len(tensor[0]), batch_size):
        yield tuple(x[i:i+batch_size] for x in tensor)


