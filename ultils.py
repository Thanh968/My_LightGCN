import pandas as pd
from torch import nn, optim
from model import PairWiseModel


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