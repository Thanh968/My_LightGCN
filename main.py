import pandas as pd
import numpy as np
from data import BasicDataset ,SampleGenerator
import procedure
from model import LightGCN
import procedure
import ultils 

config = {
    'keep_prob': 0.6,
    'latent_dim_rec': 64,
    'lightGCN_n_layers': 3,
    'A_split': False,
    'decay': 1e-4,
    'lr': 0.001
}

rating = pd.read_csv('ratings.dat', sep=',', header=None, names=['uid','mid', 'rating', 'timestamp'], engine='python')
# truyen dataframe vao dataloader
user_id = rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
rating = pd.merge(rating, user_id, on=['uid'], how='left')
item_id = rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
rating = pd.merge(rating, item_id, on=['mid'], how='left')
rating = rating[['userId', 'itemId', 'rating', 'timestamp']]

sample_generator = SampleGenerator(rating)
validate_data = sample_generator.validate_data
test_data = sample_generator.test_data

Recmodel = LightGCN(config=config, dataset= sample_generator)
bpr = ultils.BPR_Loss(Recmodel, config)
procedure.BPR_train(sample_generator, Recmodel, bpr)



