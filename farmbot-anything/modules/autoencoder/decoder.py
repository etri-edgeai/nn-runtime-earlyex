import torch
import torch.nn as nn
import torch.nn.functional as F



class PCDDecoder(nn.Module):
    def __init__(self, embedding_dims= 512, pcd_samples = 2048):
        super(PCDDecoder, self).__init__()
        self.embedding_dims = embedding_dims
        self.pcd_samples    = pcd_samples
        self.fc1            = nn.Linear(embedding_dims, 512)
        self.fc2            = nn.Linear(512, 1024)
        self.fc3            = nn.Linear(1024, 3 * pcd_samples)


    def forward(self, x):
        """
        Args:
            x: (B, embedding_dims)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # (B, 3 * pcd_samples)
        # x = x.view(-1,  3, self.pcd_samples) # (B, 3, pcd_samples)
        x = x.view(-1, self.pcd_samples, 3)
        return x
