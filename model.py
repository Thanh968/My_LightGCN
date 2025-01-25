from data import BasicDataset
import torch
import numpy as np
from torch import nn

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).init()

    def getUsersRating(self, users, items):
        raise NotImplementedError
    
class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCN,self).__init__()
        self.config: dict = config
        self.dataset: BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(num_embeddings = self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings = self.num_items, embedding_dim = self.latent_dim)
        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std = 0.1)

        self.activate_function = torch.nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

    def computer(self):
        user_weight = self.embedding_user.weight
        item_weight = self.embedding_item.weight
        all_em = torch.cat([user_weight, item_weight])

        embs = [all_em]
        graph = self.Graph

        for layer in range(self.n_layers):
            all_em = torch.sparse.mm(graph, all_em)
            embs.append(all_em)

        embs = torch.stack(embs, dim=1)

        result = torch.mean(embs, dim=1)
        users, items = torch.split(result, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users, items):
        all_users, all_items = self.computer()
        user_emb = all_users[users]
        item_emb = all_items[items]
        ratings = self.activate_function(torch.matmul(user_emb, item_emb.t()))
        return ratings

    

        

