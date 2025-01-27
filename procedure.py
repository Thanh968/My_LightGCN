from data import BasicDataset
from model import BasicModel
import ultils
import torch

def BPR_train(dataset: BasicDataset, recmodel: BasicModel, loss_class: ultils.BPR_Loss):
    # gan mo hinh goi y
    Recmodel = recmodel
    # chuyen mo hinh ve che do huan luyen
    Recmodel.train()
    # gan lop tinh loi
    bpr: ultils.BPR_Loss = loss_class
    # Lay mau huan luyen
    S = ultils.UniformSample_original_python(dataset=dataset)
    # lay cac tensor user, pos, neg
    users = torch.Tensor(S[:, 0]).long()
    pos = torch.Tensor(S[:, 1]).long()
    neg = torch.Tensor(S[:, 2]).long()

    # chuyen cac tensor den gpu
    users = users.to('cuda')
    pos = pos.to('cuda')
    neg = neg.to('cuda')

    # tron du lieu
    users, pos, neg = ultils.shuffle(users, pos, neg)

    # tinh tong so luong batch
    total_batch = len(users) // 94 + 1
    avg_loss = 0.

    for i, (user_batch, pos_batch, neg_batch) in enumerate(ultils.minibatch(users, pos, neg, batch_size = 94)):
        cri = bpr.stageOne(user_batch, pos_batch, neg_batch)
        avg_loss += cri

    avg_loss /= total_batch
    print(f"Loss: {avg_loss}")




