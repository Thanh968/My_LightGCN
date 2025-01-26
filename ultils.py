import pandas as pd
from torch import nn, optim
from model import PairWiseModel
from data import BasicDataset
import numpy as np
import random
import torch


def getRequiredFields(dataframe, required_fields):
    result = dataframe[required_fields]
    return result;

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
    num_train_users = dataset.trainDataSize
    sample_users = np.random.choice(np.array(range(dataset.num_users), num_train_users, replace = False))
    allPos = dataset.allPos
    S = []

    for i, user in enumerate(sample_users):
        # All Pos chỉ build tren tap train nen khong sao
        userPosItems = allPos[user]
        if len(userPosItems) == 0:
            continue
        selected_pos_item_index = np.random.randint(0, len(userPosItems))
        selected_pos_item = allPos[selected_pos_item_index]

        userPosItems = set(userPosItems)
        userNegItems = set(dataset.num_items) - userPosItems

        selected_neg_item = random.choice(list(userNegItems))

        S.append([user, selected_pos_item, selected_neg_item])

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

    if (length_set != 1):
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

