from CF_SVD import SVD_CF
from CF_item_based import ItemBasedCF
from CF_user_based import UserBasedCF
from TwoTowerModel import TwoTowerModel
from dataset_processor import DatasetProcessor
from dataset_statistics import DatasetStatistics
import pandas as pd

# from evaluator import EvaluateMetrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sentence_transformers import SentenceTransformer

from evaluator import EvaluateMetrics


def main():
    # -------------------------------------------------------------------------------------------------------------
    ############################################ AMAZON BEAUTY DATASET ############################################
    # -------------------------------------------------------------------------------------------------------------

    # file_path = 'datasets/Beauty/meta_All_Beauty.jsonl/meta_All_Beauty.jsonl'
    #
    # df = pd.read_json(file_path, lines=True)
    # # Set options to display all rows
    # pd.set_option('display.max_rows', len(df))
    #
    # # Display the DataFrame
    # print(df)

    # import json
    #
    # file = 'datasets/Beauty/All_Beauty.jsonl/All_Beauty.jsonl'
    # with open(file, 'r') as fp:
    #     for line in fp:
    #         print(json.loads(line.strip()))

    # Reset the option to default after displaying
    # pd.reset_option('display.max_rows')

    # For the JSONL file
    # fields = ['title', 'rating', 'text', 'parent_asin', 'timestamp', 'helpful_vote', 'verified_purchase', 'user_id', 'asin'] # for all beauty
    # fields = ['title', 'average_rating', 'rating_number', 'parent_asin', 'features', 'description', 'price', 'store', 'categories', 'details'] # for beauty metadata

    # # Amazon Beauty Dataset
    # processor = DatasetProcessor('datasets/Beauty/meta_All_Beauty.jsonl/meta_All_Beauty.jsonl', fields, 'beauty')
    # # Load Data
    # processor.load_data()
    # # Show Dataset Statistics
    # dataset_statistics = DatasetStatistics(processor.data)
    # dataset_statistics.show_head()
    # Remove Duplicated Rows
    # dataset_statistics.show_duplicates()
    # dataset_statistics.drop_duplicates()
    # Show Distribution Plot
    # dataset_statistics.plot_distribution()
    # Remove data
    # processor.filter_data(min_ratings=3)
    # DatasetStatistics(processor.data)

    # # Split to trainset and testset
    # processor.split_data()
    # processor.save_data()
    #
    # # Load Processed Dataset
    # beauty_data = pd.read_csv('processed_dataset/Amazon-Beauty/dataset_full_trimmed_beauty.csv')
    # beauty_trainset = pd.read_csv('processed_dataset/Amazon-Beauty/dataset_train_trimmed_beauty.csv')
    # beauty_testset = pd.read_csv('processed_dataset/Amazon-Beauty/dataset_test_trimmed_beauty.csv')
    #
    # k = 5

    # # -------------------------------------------------------------------------------------------------------------
    # # Run User-Based Collaborative Filtering
    # # -------------------------------------------------------------------------------------------------------------
    # cf = UserBasedCF(beauty_trainset)
    # recommendations_beauty = cf.generate_recommendations(beauty_testset, k)
    # # Evaluate
    # evaluator = EvaluateMetrics(beauty_testset)
    # results = evaluator.evaluate_recommendations(recommendations_beauty, k)
    # print(f"NDCG@{k}: {results['NDCG@k']:.4f}")
    # print(f"MRR@{k}: {results['MRR@k']:.4f}")
    # print(f"HR@{k}: {results['HR@k']:.4f}")
    # #
    # # -------------------------------------------------------------------------------------------------------------
    # # Run Item-Based Collaborative Filtering
    # # -------------------------------------------------------------------------------------------------------------
    # cf = ItemBasedCF(beauty_trainset)
    # recommendations_beauty = cf.generate_recommendations(beauty_testset, k)
    # # Evaluate
    # evaluator = EvaluateMetrics(beauty_testset)
    # results = evaluator.evaluate_recommendations(recommendations_beauty, k)
    # print(f"NDCG@{k}: {results['NDCG@k']:.4f}")
    # print(f"MRR@{k}: {results['MRR@k']:.4f}")
    # print(f"HR@{k}: {results['HR@k']:.4f}")
    #
    # # -------------------------------------------------------------------------------------------------------------
    # ### Run SVD Collaborative Filtering
    # # -------------------------------------------------------------------------------------------------------------
    # svd = SVD_CF(train_df=beauty_trainset, full_df=beauty_data)
    # svd.fit()
    # svd.predict()
    # recommendations = svd.generate_recommendations(k)
    # # Evaluate
    # evaluator = EvaluateMetrics(beauty_testset)
    # results = evaluator.evaluate_recommendations_svd(recommendations, 5)
    # print(f"NDCG@{5}: {results['NDCG@k']:.4f}")
    # print(f"MRR@{5}: {results['MRR@k']:.4f}")
    # print(f"HR@{5}: {results['HR@k']:.4f}")

    # -------------------------------------------------------------------------------------------------------------
    ############################################ MOVIELENS 1M DATASET ############################################
    # -------------------------------------------------------------------------------------------------------------

    # # For the .dat file
    fields = ['rating', 'user_id', 'item_id', 'timestamp']  # for ratings
    # fields = ['user_id', 'gender', 'age', 'occupation', 'zip_code']  # for users
    # fields = ['item_id', 'title', 'genres']  # for movies


    # MovieLens Dataset
    dat_file_path = 'datasets/ml-1m/ml-1m/ratings.dat'
    processor_dat = DatasetProcessor(dat_file_path, fields, 'movielens')
    # Load Data
    processor_dat.load_data()
    processor_dat.sort_by_column(column_name="timestamp")
    # Show Dataset Statistics
    stats = DatasetStatistics(processor_dat.data)
    # Remove Duplicated Rows
    stats.show_duplicates()
    stats.drop_duplicates()
    # processor_dat.split_data()
    # stats.plot_distribution()

    # gender_mapping = {
    #     "F": "Female",
    #     "M": "Male"
    # }
    #
    # age_mapping = {
    #     1: "Under 18",
    #     18: "18-24",
    #     25: "25-34",
    #     35: "35-44",
    #     45: "45-49",
    #     50: "50-55",
    #     56: "56+"
    # }
    #
    # occupation_mapping = {
    #     0: "other or not specified",
    #     1: "academic/educator",
    #     2: "artist",
    #     3: "clerical/admin",
    #     4: "college/grad student",
    #     5: "customer service",
    #     6: "doctor/health care",
    #     7: "executive/managerial",
    #     8: "farmer",
    #     9: "homemaker",
    #     10: "K-12 student",
    #     11: "lawyer",
    #     12: "programmer",
    #     13: "retired",
    #     14: "sales/marketing",
    #     15: "scientist",
    #     16: "self-employed",
    #     17: "technician/engineer",
    #     18: "tradesman/craftsman",
    #     19: "unemployed",
    #     20: "writer"
    # }
    # processor_dat.map_column_values('occupation', occupation_mapping)
    # processor_dat.map_column_values('gender', gender_mapping)
    # processor_dat.map_column_values('age', age_mapping)

    # # Split to trainset and testset
    processor_dat.split_data()
    processor_dat.save_data(output_dir='.', prefix='ml_1m')
    #
    # k = 5
    # # # Load Processed Dataset
    # movielens_data = pd.read_csv('processed_dataset/MovieLens-1M/ratings/ml_1m_full_movielens.csv')
    # movielens_trainset = pd.read_csv('processed_dataset/MovieLens-1M/ratings/ml_1m_train_movielens.csv')
    # movielens_testset = pd.read_csv('processed_dataset/MovieLens-1M/ratings/ml_1m_test_movielens.csv')
    # #
    # # # -------------------------------------------------------------------------------------------------------------
    # # ### Run User-Based Collaborative Filtering
    # # # -------------------------------------------------------------------------------------------------------------
    # cf = UserBasedCF(movielens_trainset)
    # recommendations_movielens = cf.generate_recommendations(movielens_testset[:5], k)
    # print(recommendations_movielens)
    # # Evaluate
    # evaluator = EvaluateMetrics(movielens_testset)
    # results = evaluator.evaluate_recommendations(recommendations_movielens, k)
    # print(f"NDCG@{k}: {results['NDCG@k']:.4f}")
    # print(f"MRR@{k}: {results['MRR@k']:.4f}")
    # print(f"HR@{k}: {results['HR@k']:.4f}")

    # -------------------------------------------------------------------------------------------------------------
    ### Run Item-Based Collaborative Filtering
    # -------------------------------------------------------------------------------------------------------------
    # cf = ItemBasedCF(movielens_trainset)
    # recommendations_movielens = cf.generate_recommendations(movielens_testset, k)
    # # Evaluate
    # evaluator = EvaluateMetrics(movielens_testset)
    # results = evaluator.evaluate_recommendations(recommendations_movielens, k)
    # print(f"NDCG@{k}: {results['NDCG@k']:.4f}")
    # print(f"MRR@{k}: {results['MRR@k']:.4f}")
    # print(f"HR@{k}: {results['HR@k']:.4f}")

    # # -------------------------------------------------------------------------------------------------------------
    # ### Run SVD Collaborative Filtering
    # # -------------------------------------------------------------------------------------------------------------
    # svd = SVD_CF(train_df=movielens_trainset, full_df=movielens_data)
    # svd.fit()
    # svd.predict()
    # recommendations = svd.generate_recommendations(k)
    # # Evaluate
    # evaluator = EvaluateMetrics(movielens_testset)
    # results = evaluator.evaluate_recommendations_svd(recommendations, k)
    # print(f"NDCG@{k}: {results['NDCG@k']:.4f}")
    # print(f"MRR@{k}: {results['MRR@k']:.4f}")
    # print(f"HR@{k}: {results['HR@k']:.4f}")


if __name__ == "__main__":
    main()
