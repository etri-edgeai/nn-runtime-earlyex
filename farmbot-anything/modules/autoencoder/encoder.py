import torch
import torch.nn as nn
import torch.nn.functional as F

class PCDEncoder(nn.Module):
    def __init__(self, embedding_dims= 1024):
        super(PCDEncoder, self).__init__()
        self.embedding_dims = embedding_dims
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(256, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, embedding_dims, 1)
        self.fc    = nn.Linear(embedding_dims, embedding_dims)
        self.residual = nn.Conv1d(3, embedding_dims, 1)  # Residual connection

    def forward(self, x):
        """
        Args:
            x: (B, 3, N)
        """
        pcd_num = x.shape[-2]
        x = x.view(-1, 3, pcd_num)
        x_initial = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        global_feat = F.adaptive_max_pool1d(x, 1)
        x = torch.cat([x, global_feat.repeat(1, 1, pcd_num)], dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) # (B, 512, N)
        x = F.relu(self.conv5(x)) # (B, embedding_dims, N)
        x_residual = self.residual(x_initial)  # Apply the residual connection
        x = F.adaptive_avg_pool1d(x + x_residual, 1).squeeze() # (B, embedding_dims)
        x = self.fc(x) # (B, embedding_dims)
        return x