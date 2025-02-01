from data import BasicDataset
from model import BasicModel, LightGCN
import ultils
import torch
import copy
from metrics import MetronATK

metron = MetronATK(10)


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
    return avg_loss

def evaluate(Recmodel: LightGCN ,evaluate_data):
    test_users, test_items = evaluate_data[0], evaluate_data[1]
    negative_users, negative_items = evaluate_data[2], evaluate_data[3]

    ground_truth = [0] * 100
    ground_truth[0] = 1

    ground_truth = torch.FloatTensor(ground_truth)
    test_users = test_users.cuda()
    test_items = test_items.cuda()
    negative_users = negative_users.cuda()
    negative_items = negative_items.cuda()
    ground_truth = ground_truth.cuda()

    test_scores = None
    negative_scores = None

    for user in range(943):
        model = copy.deepcopy(Recmodel)
        model.eval()

        with torch.no_grad():
            test_user = test_users[user: (user + 1)]
            test_item = test_items[user: (user + 1)]
            negative_user = negative_users[user * 99: (user + 1) * 99]
            negative_item = negative_items[user * 99: (user + 1 ) * 99]

            test_score = model(test_user, test_item)
            negative_score = model(negative_user, negative_item)

            if user == 0:
                test_scores = test_score
                negative_scores = negative_score
            else:
                test_scores = torch.cat((test_scores, test_score))
                negative_scores = torch.cat((negative_scores, negative_score))

            
    test_users = test_users.cpu()
    test_items = test_items.cpu()
    test_scores = test_scores.cpu()
    negative_users = negative_users.cpu()
    negative_items = negative_items.cpu()
    negative_scores = negative_scores.cpu()

    metron.subject = [test_users.data.view(-1).tolist(),
                    test_items.data.view(-1).tolist(),
                    test_scores.data.view(-1).tolist(),
                    negative_users.data.view(-1).tolist(),
                    negative_items.data.view(-1).tolist(),
                    negative_scores.data.view(-1).tolist()]
    
    hit_ratios, ndcgs = metron.hit_ratio_k(), metron.ndcg_k()
    return hit_ratios, ndcgs






