import torch
from torch import nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.query_transform = nn.Linear(input_dim, input_dim)
        self.key_transform = nn.Linear(input_dim, input_dim)
        self.value_transform = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value):
        # Transform inputs
        query = self.query_transform(query)
        key = self.key_transform(key)
        value = self.value_transform(value)
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended_values = torch.matmul(attention_weights, value)
        return attended_values
