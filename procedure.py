from data import BasicDataset
from model import BasicModel
import ultils
import torch

def BPR_train(dataset: BasicDataset, recmodel: BasicModel, loss_class: ultils.BPR_Loss):
    Recmodel = recmodel
    Recmodel.train()
    bpr: ultils.BPR_Loss = loss_class

    S = ultils.UniformSample_original_python(dataset)

    users_tensor = torch.Tensor(S[:, 0]).long()
    pos_tensor = torch.Tensor(S[:, 1]).long()
    neg_tensor = torch.Tensor(S[:, 2]).long()

    