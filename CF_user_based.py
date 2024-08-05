import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    def __init__(self, trainset):
        self.trainset = trainset
        self.user_id_map = None
        self.item_id_map = None
        self.sparse_user_item = None
        self.user_item_matrix = None
        self.sparse_item_similarity = None
        self._prepare_data()

    def _prepare_data(self):
        self._create_mappings()
        self._build_item_user_matrix()
        self._build_user_similarity_matrix()

    def _create_mappings(self):
        self.user_ids = self.trainset['user_id'].astype('category').cat.codes
        self.item_ids = self.trainset['item_id'].astype('category').cat.codes

        self.user_id_map = pd.Series(self.trainset['user_id'].astype('category').cat.categories)
        self.item_id_map = pd.Series(self.trainset['item_id'].astype('category').cat.categories)

    def _build_item_user_matrix(self):
        self.sparse_user_item = csr_matrix((self.trainset['rating'], (self.user_ids, self.item_ids)))
        print('Shape of Sparse User-Item matrix:', self.sparse_user_item.shape)

        self.user_item_matrix = pd.DataFrame.sparse.from_spmatrix(self.sparse_user_item, index=self.user_id_map,
                                                                  columns=self.item_id_map)

    def _build_user_similarity_matrix(self):
        self.sparse_user_similarity = cosine_similarity(self.sparse_user_item, dense_output=False)
        print("Shape of Sparse User Similarity matrix:", self.sparse_user_similarity.shape)

        self.user_similarity_df = pd.DataFrame.sparse.from_spmatrix(self.sparse_user_similarity, index=self.user_id_map,
                                                                    columns=self.user_id_map)

    # def get_recommendations(self, user_id, k):
    #
    #     if user_id not in self.user_similarity_df.index:
    #         return pd.Series()
    #     # Get the similarity scores for the user and filter out users with similarity <= 0
    #     similar_users = self.user_similarity_df[user_id][self.user_similarity_df[user_id] > 0].sort_values(
    #         ascending=False)
    #     # print(similar_users)
    #
    #     if similar_users.empty:
    #         return pd.Series()  # Return an empty series if no similar users with similarity > 0
    #
    #     # Get indices of similar users
    #     similar_users_indices = similar_users.index
    #
    #     # Create a sparse matrix of user ratings filtered by similar users
    #     user_ratings_sparse = self.user_item_matrix.loc[similar_users_indices].to_numpy()
    #     user_similarities = similar_users.to_numpy()
    #
    #     # Calculate weighted ratings
    #     weighted_ratings = user_ratings_sparse.T.dot(user_similarities) / user_similarities.sum()
    #
    #     # Normalize weighted ratings to be between 1 and 5
    #     min_rating = 1
    #     max_rating = 5
    #     weighted_ratings = min_rating + (max_rating - min_rating) * (weighted_ratings - weighted_ratings.min()) / (
    #             weighted_ratings.max() - weighted_ratings.min())
    #
    #     # Convert the weighted ratings back to a Series with item IDs as index
    #     weighted_ratings_series = pd.Series(weighted_ratings, index=self.user_item_matrix.columns)
    #
    #     # Exclude items the user has already rated
    #     already_rated = self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index
    #     recommendations = weighted_ratings_series.drop(already_rated).sort_values(ascending=False).head(k)
    #
    #     return recommendations

    def get_recommendations(self, user_id, k):
        if user_id not in self.user_similarity_df.index:
            return pd.Series()

        # Get the similarity scores for the user and filter out users with similarity <= 0
        similar_users = self.user_similarity_df[user_id][self.user_similarity_df[user_id] > 0].sort_values(
            ascending=False)

        if similar_users.empty:
            return pd.Series()  # Return an empty series if no similar users with similarity > 0

        # Get indices of similar users
        similar_users_indices = similar_users.index

        # Create a sparse matrix of user ratings filtered by similar users
        user_ratings_sparse = self.user_item_matrix.loc[similar_users_indices].to_numpy()
        user_similarities = similar_users.to_numpy()

        # Calculate weighted ratings
        weighted_ratings = user_ratings_sparse.T.dot(user_similarities) / user_similarities.sum()

        # Normalize weighted ratings to be between 1 and 5
        min_rating = 1
        max_rating = 5
        weighted_ratings = min_rating + (max_rating - min_rating) * (weighted_ratings - weighted_ratings.min()) / (
                weighted_ratings.max() - weighted_ratings.min())

        # Convert the weighted ratings back to a Series with item IDs as index
        weighted_ratings_series = pd.Series(weighted_ratings, index=self.user_item_matrix.columns)

        # Exclude items the user has already rated
        already_rated = self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index
        recommendations = weighted_ratings_series.drop(already_rated).sort_values(ascending=False).head(k)

        return recommendations

    def generate_recommendations(self, testset, k):
        recommendations = {}
        users = testset['user_id'].unique()

        for i, user in enumerate(users, start=1):
            recommendations[user] = self.get_recommendations(user, k)
            print(f"Processed {i}/{len(users)} users")

        return recommendations
