import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetStatistics:
    def __init__(self, df):
        self.df = df
        self.head = None
        self.info = None
        self.missing_values = None
        self.duplicates = None
        self.unique_users = None
        self.unique_items = None
        self.top_users_data = None
        self.density = None

        # self.show_head()
        # self.show_tail()
        self.show_info()
        # self.show_missing_values()
        # self.show_duplicates()
        # self.drop_duplicates()
        # self.show_unique_counts()
        # self.top_users()
        # self.rating_density()

    def show_head(self):
        self.head = self.df.head()
        print("Head of the dataset:")
        print(self.head)

    def show_info(self):
        print("Info of the dataset:")
        self.info = self.df.info()

    def show_missing_values(self):
        self.missing_values = self.df.isnull().sum()
        print("Missing values in the dataset:")
        print(self.missing_values)

    def show_duplicates(self):
        rating_combination = ['user_id', 'item_id']
        self.duplicates = self.df[self.df.duplicated(subset=rating_combination, keep=False)].sort_values(
            rating_combination)
        print("Duplicated ratings:")
        print(self.duplicates.head())

    def drop_duplicates(self):
        self.df.drop_duplicates(subset=['user_id', 'item_id', 'rating'], inplace=True)

    def show_unique_counts(self):
        self.unique_users = self.df['user_id'].nunique()
        self.unique_items = self.df['item_id'].nunique()
        print(f"Count of unique Users    : {self.unique_users}")
        print(f"Count of unique Products : {self.unique_items}")

    def top_users(self, top_n=10):
        self.top_users_data = self.df['user_id'].value_counts().rename_axis('user_id').reset_index(name='# ratings')
        print(f"Top {top_n} users based on ratings:")
        print(self.top_users_data.head(top_n))

    def rating_density(self):
        total_observed_ratings = len(self.df)
        possible_num_of_ratings = self.df['user_id'].nunique() * self.df['item_id'].nunique()
        self.density = (total_observed_ratings / possible_num_of_ratings) * 100
        print(f'Total observed ratings in the dataset  : {total_observed_ratings}')
        print(f'Total ratings possible for the dataset : {possible_num_of_ratings}')
        print(f'Density of the dataset                 : {self.density:.5f}%')

    def users_with_few_ratings(self, threshold=5):
        user_ratings_count = self.df['user_id'].value_counts()
        users_with_less_than_threshold_ratings = user_ratings_count[user_ratings_count < threshold]
        number_of_users_with_less_than_threshold_ratings = users_with_less_than_threshold_ratings.count()
        print(
            f"Number of users with fewer than {threshold} ratings: {number_of_users_with_less_than_threshold_ratings}")

    def plot_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['rating'], bins=20, kde=True)
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.show()
