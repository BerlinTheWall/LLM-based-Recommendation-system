from surprise import Dataset, Reader, SVD
from collections import defaultdict


class SVD_CF:
    def __init__(self, train_df, full_df):
        self.reader = Reader(rating_scale=(full_df['rating'].min(), full_df['rating'].max()))
        self.full_df = self._prepare_data(full_df)
        self.train_df = self._prepare_train_data(train_df)
        self.predictions = None
        self.model = SVD()

    def _prepare_data(self, input_data):
        data = Dataset.load_from_df(input_data[['user_id', 'item_id', 'rating']], self.reader)
        return data

    def _prepare_train_data(self, input_data):
        train_data = Dataset.load_from_df(input_data[['user_id', 'item_id', 'rating']], self.reader)
        return train_data.build_full_trainset()
    def fit(self):
        self.model.fit(self.full_df.build_full_trainset())

    def predict(self):
        self.predictions = self.model.test(self.train_df.build_anti_testset())

    def generate_recommendations(self, k):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            top_n[uid].append((iid, est))

        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:k]

        return top_n
