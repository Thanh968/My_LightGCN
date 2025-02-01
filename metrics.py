import math
import pandas as pd

class MetronATK:
    def __init__(self, top_k):
        self.__top_k = top_k
        self.__subject = None

    @property
    def top_k(self):
        return self.__top_k

    @top_k.setter
    def top_k(self, top_k):
        if top_k <= 0:
            raise ValueError("top_k must be a positive number")

        self.__top_k = top_k

    @property
    def subject(self):
        result = self.__subject
        return result

    @subject.setter
    def subject(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        negative_users, negative_items, negative_scores = subjects[3], subjects[4], subjects[5]

        test = pd.DataFrame({
            'userId': test_users,
            'test_item': test_items,
            'test_score': test_scores
        });

        full = pd.DataFrame({
            'userId': negative_users + test_users,
            'itemId': negative_items + test_items,
            'score': negative_scores + test_scores
        })

        full = pd.merge(full, test, on='userId', how='left')
        full['rank'] = full.groupby('userId')['score'].rank(method='first', ascending=False)

        full.sort_values(['userId', 'rank'], inplace=True)
        self.__subject = full

    def hit_ratio_k(self):
        full, top_k = self.__subject, self.__top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['itemId']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['userId'].nunique()

    def ndcg_k(self):
        full, top_k = self.__subject, self.__top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['itemId']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['userId'].nunique()
