#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from libreco.data import DatasetPure
from libreco.algorithms import UserCF


# In[47]:


train_ratings = pd.read_csv('processed_dataset/MovieLens-1M/ratings/ml_1m_train_movielens.csv')
val_ratings = pd.read_csv('processed_dataset/MovieLens-1M/ratings/ml_1m_val_movielens.csv')
test_ratings = pd.read_csv('processed_dataset/MovieLens-1M/ratings/ml_1m_test_movielens.csv')

movies = pd.read_csv('processed_dataset/MovieLens-1M/movies/movies_movielens.csv')


# In[3]:


train_ratings.rename(columns={'user_id': 'user', 'item_id': 'item', 'rating': 'label', 'timestamp': 'time'},
                     inplace=True)
val_ratings.rename(columns={'user_id': 'user', 'item_id': 'item', 'rating': 'label', 'timestamp': 'time'}, inplace=True)
test_ratings.rename(columns={'user_id': 'user', 'item_id': 'item', 'rating': 'label', 'timestamp': 'time'},
                    inplace=True)


# In[4]:


train_data, data_info = DatasetPure.build_trainset(train_ratings)
eval_data = DatasetPure.build_evalset(val_ratings)
test_data = DatasetPure.build_testset(test_ratings)
print(data_info)


# In[5]:


train_ratings = train_ratings.sort_values(by='time')
val_ratings = val_ratings.sort_values(by='time')
test_ratings = test_ratings.sort_values(by='time')


# In[6]:


user_cf = UserCF(task="ranking", data_info=data_info, k_sim=200, sim_type="cosine", mode='invert')


# In[7]:


# Training the model
user_cf.fit(train_data, verbose=2, eval_data=eval_data, k=5,
            metrics=["loss", "roc_auc", "precision", "recall", "ndcg"], neg_sampling=True)


# ### Two-Tower Model

# In[11]:


import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
import numpy as np
from pytorch_lightning.callbacks import Callback
import pandas as pd


# In[12]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name(0)


# In[13]:


full_ratings = pd.read_csv('processed_dataset/MovieLens-1M/ratings/ml_1m_full_movielens.csv')
train_ratings = pd.read_csv('processed_dataset/MovieLens-1M/ratings/ml_1m_train_movielens.csv')
val_ratings = pd.read_csv('processed_dataset/MovieLens-1M/ratings/ml_1m_val_movielens.csv')
test_ratings = pd.read_csv('processed_dataset/MovieLens-1M/ratings/ml_1m_test_movielens.csv')

movies = pd.read_csv('processed_dataset/MovieLens-1M/movies/movies_movielens_modified.csv')
users = pd.read_csv('processed_dataset/MovieLens-1M/users/users_movielens_modified.csv')


# In[431]:


movies['movie_features'] = '[MOVIE_DETAIL] title: ' + movies['title'] + ' [SEP] genres: ' + movies['genres']


# In[432]:


# Create a dictionary for fast lookup
movie_features_dict = movies.set_index('item_id')['movie_features'].to_dict()

# Create lists of user and item texts
item_texts = [movie_features_dict[movieId] for movieId in full_ratings['item_id'].unique()]

# Create a mapping from userId and movieId to indices
movie_id_to_idx = {movieId: idx for idx, movieId in enumerate(full_ratings['item_id'].unique())}

# Map userId and movieId in ratings_df to indices
train_ratings['movie_idx'] = train_ratings['item_id'].map(movie_id_to_idx)

# Map userId and movieId in ratings_val to indices
val_ratings['movie_idx'] = val_ratings['item_id'].map(movie_id_to_idx)

# Map userId and movieId in ratings_test to indices
test_ratings['movie_idx'] = test_ratings['item_id'].map(movie_id_to_idx)

# Extract user indices, item indices, and ratings
train_item_indices = torch.LongTensor(train_ratings['movie_idx'].values).to(device)
train_labels = torch.FloatTensor(train_ratings['rating'].values).to(device)

# Extract user indices, item indices, and ratings for validation
val_item_indices = torch.LongTensor(val_ratings['movie_idx'].values).to(device)
val_labels = torch.FloatTensor(val_ratings['rating'].values).to(device)

# Extract user indices, item indices, and ratings for test
test_item_indices = torch.LongTensor(test_ratings['movie_idx'].values).to(device)
test_labels = torch.FloatTensor(test_ratings['rating'].values).to(device)


# In[16]:


class TwoTowerModel(pl.LightningModule):
    def __init__(self, user_model_name, item_model_name, embedding_size=384):
        super(TwoTowerModel, self).__init__()
        self.user_model = SentenceTransformer(user_model_name)
        self.item_model = SentenceTransformer(item_model_name)

        self.user_fc = nn.Linear(embedding_size, embedding_size)
        self.item_fc = nn.Linear(embedding_size, embedding_size)

        self.criterion = nn.MSELoss()
        self.epoch_losses = {'train_loss': [], 'val_loss': []}

    def forward(self, user_text, item_text):
        user_embedding = self.user_model.encode(user_text, convert_to_tensor=True).to(device)
        item_embedding = self.item_model.encode(item_text, convert_to_tensor=True).to(device)

        user_output = self.user_fc(user_embedding)
        item_output = self.item_fc(item_embedding)

        dot_product = torch.matmul(user_output.squeeze(), item_output.T)
        dot_product = 4 * torch.sigmoid(dot_product) + 1

        return dot_product

    def training_step(self, batch, batch_idx):
        users, items, ratings = batch

        items = [item_texts[i] for i in items.tolist()]

        preds = self(users, items)

        loss = self.criterion(preds, ratings)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        users, items, ratings = batch

        items = [item_texts[i] for i in items.tolist()]

        preds = self(users, items)

        loss = self.criterion(preds, ratings)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-5)


class PrintLossesCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss')
        if train_loss is not None:
            pl_module.epoch_losses['train_loss'].append(train_loss.item())
            print(f"Epoch {trainer.current_epoch + 1}: Train Loss: {train_loss.item()}")

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            pl_module.epoch_losses['val_loss'].append(val_loss.item())
            print(f"Epoch {trainer.current_epoch + 1}: Val Loss: {val_loss.item()}")


# In[17]:


best_model_path = './lightning_logs/movies/paraphrase-MiniLM-L12-v2/not-binarized/history_5-epochs_lr-1e-5_(occu + gen ) (new format - only genre for movies) (with header tag)/checkpoints/epoch=4-step=62320.ckpt'
# best_model = TwoTowerModel.load_from_checkpoint(best_model_path, user_model_name='paraphrase-MiniLM-L6-v2', item_model_name='paraphrase-MiniLM-L6-v2').to(device)
best_model = TwoTowerModel.load_from_checkpoint(best_model_path, user_model_name='paraphrase-MiniLM-L12-v2',
                                                item_model_name='paraphrase-MiniLM-L12-v2').to(device)


# In[18]:


def generate_last_user_texts_with_history(users, movies, ratings):
    user_histories = {user_id: [] for user_id in users['user_id'].unique()}
    last_user_texts = {}

    # Convert relevant columns to dictionaries for faster access
    user_features_dict = users.set_index('user_id').to_dict('index')
    movie_titles_dict = movies.set_index('item_id')['genres'].to_dict()

    for _, row in ratings.iterrows():
        user_id = row['user_id']
        movie_id = row['item_id']

        # Get user features
        user = user_features_dict[user_id]
        user_features = f"[USER_PROFILE] occupation: {user['occupation']} [SEP] gender: {user['gender']}"

        # Append the user's history (only the last 3 movies)
        history_movies = [movie_titles_dict[mid] for mid in user_histories[user_id][-3:]]

        history_str = ", ".join(history_movies)

        if history_str:
            combined_features = f"{user_features} [SEP] genres: {history_str}"
        else:
            combined_features = f"{user_features}"

        # Update the dictionary to keep the last text for each user
        last_user_texts[user_id] = combined_features

        # Update the user history after generating combined features
        user_histories[user_id].append(movie_id)

    return last_user_texts


# Generate the last user texts for the validation data
val_last_user_texts = generate_last_user_texts_with_history(users, movies, val_ratings)


# In[434]:


full_items_embeddings = torch.stack(
    [best_model.item_model.encode(item_text, convert_to_tensor=True) for item_text in item_texts]).to(device)


# ## Gradio Visualization

# In[437]:


import gradio as gr


# In[446]:


def CFGenerator(user_id):
    k=10
    user_id = int(user_id)
    # Check if user_id exists in the users dataframe
    if user_id not in users["user_id"].values:
        return [f"User ID {user_id} does not exist in the database."]

    recommended_items = user_cf.recommend_user(user_id, n_rec=k, filter_consumed=True)
    movie_ids = recommended_items[user_id]
    movie_titles = []
    for movie_id in movie_ids:
        title_row = movies[movies["item_id"] == movie_id]
        if not title_row.empty:
            movie_titles.append(title_row["title"].values[0])
        else:
            movie_titles.append(f"Unknown Movie ID: {movie_id}")

    # Add ranks to the movie titles with proper newline formatting
    movie_titles_string = "\n".join([f"{i + 1}. {title}" for i, title in enumerate(movie_titles)])
    return movie_titles_string


# In[447]:


def TTGenerator(input_data, input_method="User ID"):
    k=10
    try:
        if input_method == "User ID":
            # Validate user ID input
            try:
                user_id = int(input_data)
            except ValueError:
                return "Invalid User ID. Please enter a valid integer."

            # Generate recommendations using user ID
            movie_ids = get_top_n_items_with_history_unseen_items_gradio(
                best_model, user_id, n=k, input_type="userId"
            )

        elif input_method == "Manual Data Entry":
            # Validate user text input
            user_text = input_data.strip()
            if not user_text:
                return "No user data provided. Please enter valid user text."

            # Generate recommendations using manual user text
            movie_ids = get_top_n_items_with_history_unseen_items_gradio(
                best_model, user_text, n=k, input_type="userText"
            )

        else:
            return "Invalid input method. Please choose either 'User ID' or 'Manual Data Entry'."
        # Retrieve movie titles
        movie_titles = []
        for movie_id in movie_ids:
            title_row = movies[movies["item_id"] == movie_id]
            if not title_row.empty:
                movie_titles.append(title_row["title"].values[0])
            else:
                movie_titles.append(f"Unknown Movie ID: {movie_id}")

        # Add ranks to the movie titles with proper newline formatting
        movie_titles_string = "\n".join([f"{i + 1}. {title}" for i, title in enumerate(movie_titles)])
        return movie_titles_string

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"An error occurred: {str(e)}"


# In[440]:


def get_top_n_items_with_history_unseen_items_gradio(model, user_input, n, input_type="userId"):
    # Ensure the model is in evaluation mode
    model.eval()
    userId = 0
    if input_type == "userId":
        # Handle userId input
        try:
            userId = int(user_input)
        except ValueError:
            raise ValueError("Invalid userId. Please provide a valid integer.")

        if userId not in val_last_user_texts:
            raise ValueError(f"User ID {userId} does not exist in the database.")

        # Get the user text for the given userId
        user_text = val_last_user_texts[userId]
    elif input_type == "userText":
        # Handle userText input
        user_text = user_input.strip()
        if not user_text:
            raise ValueError("Invalid user text. Please provide non-empty text.")
    else:
        raise ValueError("Invalid input type. Must be 'userId' or 'userText'.")

    # Encode the user text
    user_embedding = model.user_model.encode(user_text, convert_to_tensor=True).to(device)
    # Compute the scores (dot product between user embedding and each item embedding)
    user_output = model.user_fc(user_embedding).to(device)
    item_output = model.item_fc(full_items_embeddings).to(device)
    dot_product = torch.matmul(user_output, item_output.t()).squeeze()

    if input_type == "userId":
        # Get items the user has seen in the training and validation data
        seen_items_train = train_ratings[train_ratings['user_id'] == userId]['item_id'].values
        seen_items_val = val_ratings[val_ratings['user_id'] == userId]['item_id'].values
        seen_items = set(np.concatenate((seen_items_train, seen_items_val)))

        # Get the top n + len(seen_items) item indices and their scores
        top_n_scores, top_n_indices = torch.topk(dot_product, n + len(seen_items))

        # Map indices back to item IDs
        top_n_item_ids = [list(movie_id_to_idx.keys())[list(movie_id_to_idx.values()).index(idx.item())] for idx in
                          top_n_indices]

        # Filter out seen items
        unseen_top_n_item_ids = [item for item in top_n_item_ids if item not in seen_items]

        return unseen_top_n_item_ids[:n]

    else:
        top_n_scores, top_n_indices = torch.topk(dot_product, n)
        top_n_item_ids = [list(movie_id_to_idx.keys())[list(movie_id_to_idx.values()).index(idx.item())] for idx in
                          top_n_indices]

        return top_n_item_ids[:n]


# In[448]:


def rrf_score(ranks, k=10):
    return sum([1 / (k + rank) for rank in ranks])

def combine_recommendations_with_rrf_with_weight_for_gradio(user_id, k=10, cf_weight=1, tt_weight=4, input_method="userId"):
    # Dictionary to hold the RRF scores
    combined_scores = {}

    # If input is "userId", only use Two-Tower recommendations
    if input_method == "Manual Data Entry":
        tt_recommended_item = get_top_n_items_with_history_unseen_items_gradio(
            best_model, user_id, n=k, input_type="userText"
        )

        # Calculate RRF scores from Two-Tower recommendations
        for rank, item in enumerate(tt_recommended_item, start=1):
            weighted_rank = rank * tt_weight  # Apply the weight to TT ranks
            if item not in combined_scores:
                combined_scores[item] = rrf_score([weighted_rank], k)
            else:
                combined_scores[item] += rrf_score([weighted_rank], k)

    # If input is "User ID", combine CF and TT recommendations
    else:
        # Get Collaborative Filtering recommendations
        user_id = int(user_id)
        cf_recommended_items = user_cf.recommend_user(user_id, n_rec=k, filter_consumed=True)

        # Assign ranks and calculate weighted RRF scores from CF recommendations
        for rank, item in enumerate(cf_recommended_items[user_id], start=1):
            weighted_rank = rank * cf_weight  # Apply the weight to CF ranks
            if item not in combined_scores:
                combined_scores[item] = rrf_score([weighted_rank], k)
            else:
                combined_scores[item] += rrf_score([weighted_rank], k)

        # Get Two-Tower recommendations
        tt_recommended_item = get_top_n_items_with_history_unseen_items_gradio(
            best_model, user_id, n=k, input_type="userId"
        )
        # print(movieTitleOutput(tt_recommended_item))

        # Assign ranks and calculate weighted RRF scores from TT recommendations
        for rank, item in enumerate(tt_recommended_item, start=1):
            weighted_rank = rank * tt_weight  # Apply the weight to TT ranks
            if item not in combined_scores:
                combined_scores[item] = rrf_score([weighted_rank], k)
            else:
                combined_scores[item] += rrf_score([weighted_rank], k)

    # Sort the items based on their RRF scores in descending order
    sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    # Retrieve the top N item IDs
    top_items = [item for item, score in sorted_items[:k]]

    # Get the movie titles for the top items
    movie_titles = []
    for movie_id in top_items:
        title_row = movies[movies["item_id"] == movie_id]
        if not title_row.empty:
            movie_titles.append(title_row["title"].values[0])
        else:
            movie_titles.append(f"Unknown Movie ID: {movie_id}")

    # Add ranks to the movie titles with proper newline formatting
    movie_titles_string = "\n".join([f"{i + 1}. {title}" for i, title in enumerate(movie_titles)])

    return movie_titles_string


# In[449]:


# Function to show user preferences dynamically
def show_user_preferences(user_id):
    try:
        user_id = int(user_id)  # Ensure user_id is an integer
        # Lookup user preferences from the DataFrame
        user_row = users[users["user_id"] == user_id]
        if not user_row.empty:
            preferences = (
                f"Occupation: {user_row['occupation'].values[0]}, "
                f"Age: {user_row['age'].values[0]}, "
                f"Gender: {user_row['gender'].values[0]}"
            )
            # Update the field to visible and return the preferences
            return gr.update(visible=True, value=preferences)
        else:
            # User not found
            return gr.update(visible=False, value="User not found!")
    except ValueError:
        # Handle invalid input
        return gr.update(visible=False, value="Invalid User ID!")


# In[454]:


# Create the Gradio app
with gr.Blocks() as demo:
    gr.Markdown("## Recommendation Systems Comparison")  # Heading

    # Row to hold two identical blocks
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Collaborative Filtering")  # Heading
            user_id_input = gr.Textbox(label="User ID", placeholder="Enter your ID here")
            user_preferences_output = gr.Textbox(label="User Preferences", visible=False)
            cf_recommendations_output = gr.Textbox(label="Recommendations")
            cf_generate_button = gr.Button("Generate CF Recommendations")

            cf_generate_button.click(CFGenerator, inputs=[user_id_input], outputs=[cf_recommendations_output])
            user_id_input.change(show_user_preferences, inputs=[user_id_input], outputs=[user_preferences_output])

        with gr.Column():
            gr.Markdown("## SBERT Two-Tower Model")  # Heading

            # Radio Button to Choose Input Method
            input_method_sbert = gr.Radio(
                choices=["User ID", "Manual Data Entry"],
                label="Choose Input Method",
                value="User ID"
            )

            # Fields for User ID Input
            user_id_input_sbert = gr.Textbox(label="User ID (SBERT)", visible=True, placeholder="Enter User ID")

            # Fields for Manual Data Entry
            occupation_input = gr.Textbox(label="Occupation", visible=False, placeholder="Enter Occupation")
            gender_input = gr.Textbox(label="Gender", visible=False, placeholder="Enter Gender")
            favorite_genres_input = gr.Textbox(label="Favorite Genres", visible=False, placeholder="Enter Favorite Genres (comma-separated)")

            # Output field
            tt_recommendations_output = gr.Textbox(label="Recommendations", interactive=False)
            tt_generate_button = gr.Button("Generate LLM Recommendations")

            # Dynamically show the relevant input fields based on the selected method
            input_method_sbert.change(
                lambda method: (
                    gr.update(visible=method == "User ID"),  # User ID field
                    gr.update(visible=method == "Manual Data Entry"),  # Occupation field
                    gr.update(visible=method == "Manual Data Entry"),  # Gender field
                    gr.update(visible=method == "Manual Data Entry")   # Favorite Genres field
                ),
                inputs=[input_method_sbert],
                outputs=[user_id_input_sbert, occupation_input, gender_input, favorite_genres_input]
            )

            # Handle button click for recommendations
            def generate_recommendations(user_id, occupation, gender, favorite_genres, input_method):
                try:
                    # Determine input_data based on the selected method
                    if input_method == "User ID":
                        input_data = user_id
                    else:
                        # Combine manual data into a single structure for processing
                        input_data = (
                            f"[USER_PROFILE] occupation: {occupation.strip()} [SEP] "
                            f"gender: {gender.strip()} [SEP] "
                            f"genres: {favorite_genres.strip()}"
                        )

                    # Call TTGenerator with the appropriate input
                    return TTGenerator(input_data=input_data, input_method=input_method)
                except Exception as e:
                    return f"An error occurred: {str(e)}"

            # Connect button to the generate_recommendations function
            tt_generate_button.click(
                generate_recommendations,
                inputs=[user_id_input_sbert, occupation_input, gender_input, favorite_genres_input, input_method_sbert],
                outputs=[tt_recommendations_output]
            )
        # Row for Hybrid Model
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Hybrid Model")  # Heading

            # Radio Button to Choose Input Method
            input_method_hybrid = gr.Radio(
                choices=["User ID", "Manual Data Entry"],
                label="Choose Input Method",
                value="User ID"
            )

            # Fields for User ID Input
            user_id_input_hybrid = gr.Textbox(label="User ID (Hybrid)", visible=True, placeholder="Enter User ID")

           # Fields for Manual Data Entry
            occupation_input_hybrid = gr.Textbox(label="Occupation", visible=False, placeholder="Enter Occupation")
            gender_input_hybrid = gr.Textbox(label="Gender", visible=False, placeholder="Enter Gender")
            favorite_genres_input_hybrid = gr.Textbox(label="Favorite Genres", visible=False, placeholder="Enter Favorite Genres (comma-separated)")

            # Output field
            hybrid_recommendations_output = gr.Textbox(label="Recommendations", interactive=False)
            hybrid_generate_button = gr.Button("Generate Hybrid Recommendations")

            # Dynamically show the relevant input fields based on the selected method
            input_method_hybrid.change(
                lambda method: (
                    gr.update(visible=method == "User ID"),  # User ID field
                    gr.update(visible=method == "Manual Data Entry"),  # Occupation field
                    gr.update(visible=method == "Manual Data Entry"),  # Gender field
                    gr.update(visible=method == "Manual Data Entry")   # Favorite Genres field
                ),
                inputs=[input_method_hybrid],
                outputs=[user_id_input_hybrid, occupation_input_hybrid, gender_input_hybrid, favorite_genres_input_hybrid]
            )

            # Handle button click for hybrid recommendations
            def generate_hybrid_recommendations(user_id, occupation, gender, favorite_genres, input_method):
                try:
                    # Determine input_data based on the selected method
                    if input_method == "User ID":
                        input_data = user_id
                        return combine_recommendations_with_rrf_with_weight_for_gradio(
                            user_id=input_data, input_method="User ID"
                        )
                    else:
                        # Combine manual data into a single structure
                        input_data = (
                            f"[USER_PROFILE] occupation: {occupation.strip()} [SEP] "
                            f"gender: {gender.strip()} [SEP] "
                            f"genres: {favorite_genres.strip()}"
                        )
                        return combine_recommendations_with_rrf_with_weight_for_gradio(
                            user_id=input_data, input_method="Manual Data Entry"
                        )
                except Exception as e:
                    return f"An error occurred: {str(e)}"

            # Connect button to the generate_hybrid_recommendations function
            hybrid_generate_button.click(
                generate_hybrid_recommendations,
                inputs=[user_id_input_hybrid, occupation_input_hybrid, gender_input_hybrid, favorite_genres_input_hybrid, input_method_hybrid],
                outputs=[hybrid_recommendations_output]
            )
# Launch the app
demo.launch(share=True)


# In[ ]:




