import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os


class DatasetProcessor:
    def __init__(self, file_path, fields, file_name):
        self.file_path = file_path
        self.fields = fields
        self.file_name = file_name
        self.data = None
        self.full_data_set = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def read_jsonl_to_df(self):
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # Parse each line as a JSON object
                    json_obj = json.loads(line.strip())
                    # Extract only the specified fields
                    filtered_obj = {field: json_obj.get(field) for field in self.fields}
                    data.append(filtered_obj)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

        # Create a DataFrame from the list of filtered JSON objects
        self.data = pd.DataFrame(data)
        # self.data.rename(columns={'overall': 'rating', 'reviewerID': 'user_id', 'asin': 'item_id'}, inplace=True)
        self.data.rename(columns={'asin': 'item_id'}, inplace=True)

    def read_dat_to_df(self):
        self.data = pd.read_csv(
            self.file_path, delimiter='::', header=None, engine='python', encoding="ISO-8859-1",
            names=['user_id', 'item_id', 'rating', 'timestamp']
            # names=['user_id', 'gender', 'age', 'occupation', 'zip_code']
            # names=['item_id', 'title', 'genres']
        )

    def read_csv_to_df(self):
        self.data = pd.read_csv(self.file_path)

    def load_data(self):
        file_extension = os.path.splitext(self.file_path)[-1].lower()
        if file_extension == '.jsonl':
            self.read_jsonl_to_df()
        elif file_extension == '.dat':
            self.read_dat_to_df()
        else:
            raise ValueError("Unsupported file format. Please provide a .jsonl or .dat file.")

    def split_data(self, train_size=0.8, val_size=0.1, test_size=0.1, random_state=5):
        if self.data is None:
            raise ValueError("Data not loaded. Please run load_data() first.")

        # Ensure that the sum of train_size, val_size, and test_size is 1.0
        if train_size + val_size + test_size != 1.0:
            raise ValueError("Train, validation, and test sizes must sum to 1.0")

        # Sort data by timestamp
        self.data = self.data.sort_values(by='timestamp')

        # Initialize empty DataFrames for the splits
        train_list = []
        val_list = []
        test_list = []

        # Group data by userId
        grouped = self.data.groupby('user_id')

        for userId, group in grouped:
            # Split the data for each userId
            train_data, temp_data = train_test_split(
                group,
                test_size=(val_size + test_size),
                random_state=random_state,
                shuffle=False  # Preserve timestamp order
            )

            val_data, test_data = train_test_split(
                temp_data,
                test_size=test_size / (val_size + test_size),
                random_state=random_state,
                shuffle=False  # Preserve timestamp order
            )

            train_list.append(train_data.sort_values(by='timestamp'))
            val_list.append(val_data.sort_values(by='timestamp'))
            test_list.append(test_data.sort_values(by='timestamp'))

        # Concatenate the lists to form the final datasets
        self.train_set = pd.concat(train_list).reset_index(drop=True)
        self.val_set = pd.concat(val_list).reset_index(drop=True)
        self.test_set = pd.concat(test_list).reset_index(drop=True)

    def save_data(self, output_dir='.', prefix=''):
        if self.train_set is not None and self.val_set is not None and self.test_set is not None and self.data is not None:
            full_data_file = os.path.join(output_dir, f"{prefix}_fulldata_{self.file_name}.csv")
            train_file = os.path.join(output_dir, f"{prefix}_traindata_{self.file_name}.csv")
            val_file = os.path.join(output_dir, f"{prefix}_valdata_{self.file_name}.csv")
            test_file = os.path.join(output_dir, f"{prefix}_testdata_{self.file_name}.csv")
            self.data.to_csv(full_data_file, index=False)
            self.train_set.to_csv(train_file, index=False)
            self.val_set.to_csv(val_file, index=False)
            self.test_set.to_csv(test_file, index=False)
            print(f"Full data set saved to: {full_data_file}")
            print(f"Train set saved to: {train_file}")
            print(f"Validation set saved to: {val_file}")
            print(f"Test set saved to: {test_file}")
        else:
            raise ValueError("Train, validation, or test set not created. Please run split_data() first.")

    def filter_data(self, min_ratings):
        if self.data is not None:
            unique_original = (self.data.user_id.nunique(), self.data.item_id.nunique())
            most_rated = self.data.user_id.value_counts().rename_axis('user_id').reset_index(name='# ratings')
            filtered_data = self.data[
                self.data.user_id.isin(most_rated[most_rated['# ratings'] > min_ratings].user_id)]

            print(f'# unique USERS who have rated {min_ratings} or more products :', filtered_data.user_id.nunique())
            print(f'# unique USERS dropped      :', unique_original[0] - filtered_data.user_id.nunique())
            print(f'# unique ITEMS remaining    :', filtered_data.item_id.nunique())
            print(f'# unique ITEMS dropped      :', unique_original[1] - filtered_data.item_id.nunique())
            print(f'\nFinal length of the dataset :', len(filtered_data))

            self.data = filtered_data
        else:
            raise ValueError("Data not loaded. Please run load_data() first.")

    def map_column_values(self, column_name, mapping_dict):
        if self.data is not None:
            if column_name in self.data.columns:
                self.data[column_name] = self.data[column_name].map(mapping_dict).fillna(self.data[column_name])
            else:
                raise ValueError(f"Column '{column_name}' not found in the data.")
        else:
            raise ValueError("Data not loaded. Please run load_data() first.")

    def sort_by_column(self, column_name, ascending=True):
        if self.data is not None:
            if column_name in self.data.columns:
                self.data = self.data.sort_values(by=column_name, ascending=ascending)
                print(f"Data sorted by '{column_name}' in {'ascending' if ascending else 'descending'} order.")
            else:
                raise ValueError(f"Column '{column_name}' not found in the data.")
        else:
            raise ValueError("Data not loaded. Please run load_data() first.")

    def binarize_ratings(self, threshold):
        if self.data is not None:
            if 'rating' in self.data.columns:
                self.data['rating'] = self.data['rating'].apply(lambda x: 1 if x >= threshold else 0)
                print(f"Ratings binarized with threshold {threshold}.")
            else:
                raise ValueError("Column 'rating' not found in the data.")
        else:
            raise ValueError("Data not loaded. Please run load_data() first.")
