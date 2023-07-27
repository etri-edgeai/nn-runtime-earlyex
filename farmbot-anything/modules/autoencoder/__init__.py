import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import PCDEncoder
from .decoder import PCDDecoder

class PCDAutoEncoder(nn.Module):
    def __init__(self, embedding_dims= 1024, pcd_samples=2048):
        super(PCDAutoEncoder, self).__init__()
        self.encoder = PCDEncoder(embedding_dims)
        self.decoder = PCDDecoder(embedding_dims, pcd_samples)
    def forward(self, x):
        """
        Args:
            x: (B, 3, Pcd_samples)
            emb: (B, embedding_dims)

        Returns:
            emb: (B, embedding_dims)
            output: (B, 3, pcd_samples)

        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x