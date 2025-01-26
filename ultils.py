import pandas as pd
from torch import nn, optim
from model import PairWiseModel
from data import BasicDataset
import numpy as np
import random


def getRequiredFields(dataframe, required_fields):
    result = dataframe[required_fields]
    return result;

class BPR_Loss:
    def __init__(self, recmodel : PairWiseModel, config: dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr = self.lr)

    def stageOne(self, users, items):
        # tinh loss va regularization loss
        loss, reg_loss = self.model.bpr_loss()
        # nhan regularization loss voi weight decay
        reg_loss = self.weight_decay * reg_loss
        # cong loss voi weight decay
        loss = loss + reg_loss
        # dat lai gradient cua optimizer bang 0
        self.opt.zero_grad()
        # lan truyen nguoc
        loss.backward()
        # cap nhat trong so
        self.opt.step()

        return loss.cpu().item()

def UniformSample_original_python(dataset: BasicDataset):
    num_train_users = dataset.trainDataSize
    sample_users = np.random.choice(np.array(range(dataset.num_users), num_train_users, replace = False))
    allPos = dataset.allPos
    S = []

    for i, user in enumerate(sample_users):
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