import numpy as np


class EvaluateMetrics:
    def __init__(self, test_ratings):
        self.test_ratings = test_ratings

    @staticmethod
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    @staticmethod
    def ndcg_at_k(r, k):
        dcg_max = EvaluateMetrics.dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.
        return EvaluateMetrics.dcg_at_k(r, k) / dcg_max

    @staticmethod
    def mrr_at_k(relevance_scores, k):
        for i, rel in enumerate(relevance_scores[:k]):
            if rel > 0:
                return 1 / (i + 1)
        return 0.

    @staticmethod
    def hr_at_k(relevance_scores, k):
        return int(np.any(np.asarray(relevance_scores)[:k] > 0))

    def evaluate_recommendations(self, recommendations, k):
        ndcg_scores = []
        mrr_scores = []
        hr_scores = []

        for user, user_recommendations in recommendations.items():
            # Get the true ratings for the recommended items
            true_ratings = self.test_ratings[
                (self.test_ratings['user_id'] == user) & (
                    self.test_ratings['item_id'].isin(user_recommendations.index))]
            if true_ratings.empty:
                continue

            true_ratings = true_ratings.set_index('item_id').reindex(user_recommendations.index)['rating'].fillna(0)
            relevance = true_ratings.values / 5  # Assuming ratings are from 1 to 5

            # Calculate metrics
            ndcg = self.ndcg_at_k(relevance, k)
            mrr = self.mrr_at_k(relevance, k)
            hr = self.hr_at_k(relevance, k)

            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)
            hr_scores.append(hr)

        average_ndcg = np.mean(ndcg_scores)
        average_mrr = np.mean(mrr_scores)
        average_hr = np.mean(hr_scores)

        return {
            'NDCG@k': average_ndcg,
            'MRR@k': average_mrr,
            'HR@k': average_hr
        }

    def evaluate_recommendations_svd(self, recommendations, k):
        ndcg_scores = []
        mrr_scores = []
        hr_scores = []

        for user, user_recommendations in recommendations.items():
            # Get the true ratings for the recommended items
            true_ratings = self.test_ratings[(self.test_ratings['user_id'] == user) & (
                self.test_ratings['item_id'].isin([iid for iid, _ in user_recommendations]))]
            if true_ratings.empty:
                continue

            true_ratings = true_ratings.set_index('item_id').reindex([iid for iid, _ in user_recommendations])[
                'rating'].fillna(0)
            relevance = true_ratings.values / 5  # Assuming ratings are from 1 to 5

            # Calculate metrics
            ndcg = self.ndcg_at_k(relevance, k)
            mrr = self.mrr_at_k(relevance, k)
            hr = self.hr_at_k(relevance, k)

            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)
            hr_scores.append(hr)

        average_ndcg = np.mean(ndcg_scores)
        average_mrr = np.mean(mrr_scores)
        average_hr = np.mean(hr_scores)

        return {
            'NDCG@k': average_ndcg,
            'MRR@k': average_mrr,
            'HR@k': average_hr
        }
