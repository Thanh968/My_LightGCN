import random
import pandas as pd
from copy import deepcopy
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import csr_matrix, dok_matrix, diags
import scipy as sp

def getRequiredFields(dataframe, required_fields):
    result = dataframe[required_fields]
    return result;

random.seed(0)

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def num_users(self):
        raise NotImplementedError
    
    @property
    def num_items(Self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def test_data(self):
        raise NotImplementedError
    
    @property
    def validate_data(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |0,   R|
            |R^T, 0|
        """
        raise NotImplementedError

class SampleGenerator(BasicDataset):
    def __init__(self, ratings):
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        self.preprocessed_ratings = self._binarize()
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())

        self.negatives = self._sample_negative()
        self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(self.preprocessed_ratings)

        self._num_users = len(np.unique(ratings['userId']))
        self._num_items = len(np.unique(ratings['itemId']))

        # bipartie graph
        self.UserItemNet = csr_matrix((np.ones(len(self.train_ratings.userId)), (self.train_ratings.userId, self.train_ratings.itemId)), shape=(self.num_users, self.num_items))

    def _binarize(self):
        ratings = deepcopy(self.ratings)
        # ratings['rating'][ratings['rating'] > 0] = 1.0
        ratings.loc[ratings['rating'] > 0, 'rating'] = 1.0
        return ratings

    def _sample_negative(self):
        interact_status = self.ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'interacted_items'});
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        # Convert the set to a list before sampling
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), 198))  
        interact_status['negative_samples'] = interact_status['negative_samples'].apply(lambda x: set(x))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def _split_loo(self, ratings):
        ratings['ranking_latest'] = ratings.groupby('userId')['timestamp'].rank(method='first', ascending = False)
        test_ratings = ratings[ratings['ranking_latest'] == 1]
        val_ratings = ratings[ratings['ranking_latest'] == 2]
        train_ratings = ratings[ratings['ranking_latest'] > 2]

        train_ratings = getRequiredFields(train_ratings, ['userId', 'itemId', 'rating'])
        val_ratings = getRequiredFields(val_ratings, ['userId', 'itemId', 'rating'])
        test_ratings = getRequiredFields(test_ratings, ['userId', 'itemId', 'rating'])

        return train_ratings, val_ratings, test_ratings
    
    @property
    def num_users(self):
        return self._num_users
    
    @property
    def num_items(self):
        return self._num_items
    
    @property
    def trainDataSize(self):
        result = self.train_ratings.shape[0]
        return result

    @property
    def validate_data(self):
        val_ratings = pd.merge(self.val_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        val_users, val_items, negative_users, negative_items = [], [], [] ,[]

        for row in val_ratings.itertuples():
            val_users.append(int(row.userId))
            val_items.append(int(row.itemId))

            len_of_negative_samples = len(row.negative_samples)
            list_negative_sample = list(row.negative_samples)

            for j in range(int(len_of_negative_samples / 2)):
                negative_users.append(int(row.userId))
                negative_items.append(int(list_negative_sample[j]))

        return [torch.LongTensor(val_users), torch.LongTensor(val_items), torch.LongTensor(negative_users), torch.LongTensor(negative_items)]
    
    @property
    def test_data(self):
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [] ,[]

        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))

            list_of_negative_samples = list(row.negative_samples)
            length = len(row.negative_samples)

            for j in range(int((length - 1) / 2) + 1, length):
                negative_users.append(int(row.userId))
                negative_items.append(list_of_negative_samples[j])

        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users), torch.LongTensor(negative_items)]
    
    @property
    def allPos(self):
        users = list(range(self._num_users))
        result = self.getUserPosItems(users)
        return result

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo_matrix = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo_matrix.row).long()
        col = torch.Tensor(coo_matrix.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo_matrix.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo_matrix.shape))

    def getSparseGraph(self):
        adj_matrix = dok_matrix((self.num_items + self.num_users, self.num_items + self.num_users), dtype=np.float32)
        adj_matrix = adj_matrix.tolil()
        R = self.UserItemNet.tolil()
        adj_matrix[:self.num_users,self.num_users:] = R
        adj_matrix[self.num_users:, :self.num_users] = R.T
        adj_matrix = adj_matrix.todok()

        sumrow_matrix = np.array(adj_matrix.sum(axis=1))
        d_inv = np.power(sumrow_matrix, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = diags(d_inv, offsets = 0)

        norm_adj = d_mat.dot(adj_matrix)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj =norm_adj.tocsr()

        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(torch.device('cuda'))

        return self.Graph
    
    def getUserItemFeedback(self, users, items):
        result = np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))
        return result;

    def getUserPosItems(self, users):
        pos_items = []

        for user in users:
            pos_items.append(self.UserItemNet[user].nonzero()[1])
        
        return pos_items
