import random
import pandas as pd
from copy import deepcopy
import torch
from torch.ultils.data import DataLoader, Dataset
from ultils import getRequiredFields

random.seed(0)

class SampleGenerator(object):
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

    def _binarize(self):
        ratings = deepcopy(self.ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
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
    

    