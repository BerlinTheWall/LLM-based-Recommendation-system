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

    def evaluate_recommendation_without_history(self, recommendations, k):
        ndcg_scores = []
        mrr_scores = []
        hr_scores = []

        for user_id in recommendations.keys():
            user_recommendations = recommendations[user_id]
            # Filter test ratings for the current user
            user_test_ratings = self.test_ratings[self.test_ratings['user_id'] == user_id]

            relevance_scores = []
            for item_id in user_recommendations.index:
                if item_id in user_test_ratings['item_id'].values:
                    relevance_scores.append(
                        user_test_ratings[user_test_ratings['item_id'] == item_id]['rating'].values[0] / 5)
                else:
                    relevance_scores.append(0)

            # Calculate metrics
            ndcg_scores.append(self.ndcg_at_k(relevance_scores, k))
            mrr_scores.append(self.mrr_at_k(relevance_scores, k))
            hr_scores.append(self.hr_at_k(relevance_scores, k))

        average_ndcg = np.mean(ndcg_scores)
        average_mrr = np.mean(mrr_scores)
        average_hr = np.mean(hr_scores)

        return {
            'NDCG@k': average_ndcg,
            'MRR@k': average_mrr,
            'HR@k': average_hr
        }

    def evaluate_recommendation_with_history(self, recommendations, k):
        user_metrics = {}

        for user_id, user_recommendations in recommendations:
            # Filter test ratings for the current user
            user_test_ratings = self.test_ratings[self.test_ratings['user_id'] == user_id]

            relevance_scores = []
            for item_id in user_recommendations.index:
                if item_id in user_test_ratings['item_id'].values:
                    relevance_score = user_test_ratings[user_test_ratings['item_id'] == item_id]['rating'].values[0]
                    relevance_scores.append(relevance_score / 5)  # Normalize if ratings are 1 to 5
                else:
                    relevance_scores.append(0)

            # Calculate metrics
            ndcg = self.ndcg_at_k(relevance_scores, k)
            mrr = self.mrr_at_k(relevance_scores, k)
            hr = self.hr_at_k(relevance_scores, k)

            if user_id not in user_metrics:
                user_metrics[user_id] = {
                    'ndcg': [],
                    'mrr': [],
                    'hr': []
                }

            user_metrics[user_id]['ndcg'].append(ndcg)
            user_metrics[user_id]['mrr'].append(mrr)
            user_metrics[user_id]['hr'].append(hr)

        # Aggregate metrics for each user
        ndcg_scores = [np.mean(metrics['ndcg']) for metrics in user_metrics.values()]
        mrr_scores = [np.mean(metrics['mrr']) for metrics in user_metrics.values()]
        hr_scores = [np.mean(metrics['hr']) for metrics in user_metrics.values()]

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
