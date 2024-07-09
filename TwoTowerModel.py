# import torch.nn as nn
# import torch.optim as optim
# import torch
#
#
# # Define the Two-Tower Model
# class TwoTowerModel(nn.Module):
#     def __init__(self, embedding_dim, user_embeddings, item_embeddings):
#         super(TwoTowerModel, self).__init__()
#         self.user_embedding = nn.Embedding.from_pretrained(user_embeddings, freeze=False)
#         self.item_embedding = nn.Embedding.from_pretrained(item_embeddings, freeze=False)
#         self.fc = nn.Linear(embedding_dim * 2, 1)
#
#     def forward(self, user_indices, item_indices):
#         user_embedded = self.user_embedding(user_indices)
#         item_embedded = self.item_embedding(item_indices)
#         combined = torch.cat((user_embedded, item_embedded), dim=1)
#         output = self.fc(combined).squeeze()
#         return 4 * torch.sigmoid(output) + 1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sentence_transformers import SentenceTransformer


# Define the Two-Tower Model with SBERT fine-tuning
class TwoTowerModel(nn.Module):
    def __init__(self, sbert_model_name, device):
        super(TwoTowerModel, self).__init__()
        self.user_model = SentenceTransformer(sbert_model_name).to(device)
        self.item_model = SentenceTransformer(sbert_model_name).to(device)
        self.embedding_dim = self.user_model.get_sentence_embedding_dimension()
        self.fc = nn.Linear(self.embedding_dim * 2, 1).to(device)

    def forward(self, user_texts, item_texts):
        user_embeddings = self.user_model.encode(user_texts, convert_to_tensor=True).to(device)
        item_embeddings = self.item_model.encode(item_texts, convert_to_tensor=True).to(device)
        combined = torch.cat((user_embeddings, item_embeddings), dim=1)
        output = self.fc(combined).squeeze()
        return torch.clamp(4 * torch.sigmoid(output) + 1, min=1, max=5)
