import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedCF:
    def __init__(self, trainset):
        self.trainset = trainset
        self.user_id_map = None
        self.item_id_map = None
        self.sparse_user_item = None
        self.item_user_matrix = None
        self.sparse_item_similarity = None
        self._prepare_data()

    def _prepare_data(self):
        self._create_mappings()
        self._build_item_user_matrix()
        self._build_item_similarity_matrix()

    def _create_mappings(self):
        self.user_ids = self.trainset['user_id'].astype('category').cat.codes
        self.item_ids = self.trainset['item_id'].astype('category').cat.codes

        self.user_id_map = pd.Series(self.trainset['user_id'].astype('category').cat.categories)
        self.item_id_map = pd.Series(self.trainset['item_id'].astype('category').cat.categories)

    def _build_item_user_matrix(self):
        self.sparse_user_item = csr_matrix((self.trainset['rating'], (self.user_ids, self.item_ids)))
        print('Shape of Sparse User-Item matrix:', self.sparse_user_item.shape)

        self.item_user_matrix = pd.DataFrame.sparse.from_spmatrix(self.sparse_user_item.T, index=self.item_id_map,
                                                                  columns=self.user_id_map)

    def _build_item_similarity_matrix(self):
        self.sparse_item_similarity = cosine_similarity(self.sparse_user_item.T, dense_output=False)
        print("Shape of Sparse Item Similarity matrix:", self.sparse_item_similarity.shape)

        self.item_similarity_df = pd.DataFrame.sparse.from_spmatrix(self.sparse_item_similarity, index=self.item_id_map,
                                                                    columns=self.item_id_map)

    def get_recommendations(self, user_id, k):

        if user_id not in self.item_user_matrix.columns:
            return pd.Series()

        user_ratings = self.item_user_matrix[user_id][self.item_user_matrix[user_id] > 0]

        if user_ratings.empty:
            return pd.Series()

        similar_items = pd.Series(dtype=float)

        for item, rating in user_ratings.items():
            if item in self.item_similarity_df.index:
                similarities = self.item_similarity_df[item] * rating
                similar_items = similar_items.add(similarities, fill_value=0)

        similar_items = similar_items.drop(user_ratings.index, errors='ignore')
        similar_items = similar_items.sort_values(ascending=False).head(k)

        return similar_items

    def generate_recommendations(self, testset, k):
        recommendations = {}
        users = testset['user_id'].unique()

        for i, user in enumerate(users, start=1):
            recommendations[user] = self.get_recommendations(user, k)
            print(f"Processed {i}/{len(users)} users")

        return recommendations
