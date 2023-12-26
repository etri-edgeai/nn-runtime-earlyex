# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class PCDAutoEncoder(nn.Module):
#     def __init__(self, embedding_dims= 1024, pcd_samples=2048):
#         super(PCDAutoEncoder, self).__init__()
#         self.encoder = PCDEncoder(embedding_dims)
#         self.decoder = PCDDecoder(embedding_dims, pcd_samples)
#     def forward(self, x):
#         """
#         Args:
#             x: (B, 3, Pcd_samples)
#             emb: (B, embedding_dims)

#         Returns:
#             emb: (B, embedding_dims)
#             output: (B, 3, pcd_samples)

#         """
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
# class PCDEncoder(nn.Module):
#     def __init__(self, embedding_dims= 1024):
#         super(PCDEncoder, self).__init__()
#         self.embedding_dims = embedding_dims
#         self.conv1 = nn.Conv1d(3, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(256, 256, 1)
#         self.conv4 = nn.Conv1d(256, 512, 1)
#         self.conv5 = nn.Conv1d(512, embedding_dims, 1)
#         self.fc    = nn.Linear(embedding_dims, embedding_dims)
#         self.residual = nn.Conv1d(3, embedding_dims, 1)  # Residual connection

#     def forward(self, x):
#         """
#         Args:
#             x: (B, 3, N)
#         """
#         pcd_num = x.shape[-1]
#         x_initial = x
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         global_feat = F.adaptive_max_pool1d(x, 1)
#         x = torch.cat([x, global_feat.repeat(1, 1, pcd_num)], dim=1)
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x)) # (B, 512, N)
#         x = F.relu(self.conv5(x)) # (B, embedding_dims, N)
#         x_residual = self.residual(x_initial)  # Apply the residual connection
#         x = F.adaptive_avg_pool1d(x + x_residual, 1).squeeze() # (B, embedding_dims)
#         x = self.fc(x) # (B, embedding_dims)
#         return x



# class PCDDecoder(nn.Module):
#     def __init__(self, embedding_dims= 512, pcd_samples = 2048):
#         super(PCDDecoder, self).__init__()
#         self.embedding_dims = embedding_dims
#         self.pcd_samples    = pcd_samples
#         self.fc1            = nn.Linear(embedding_dims, 512)
#         self.fc2            = nn.Linear(512, 1024)
#         self.fc3            = nn.Linear(1024, 3 * pcd_samples)


#     def forward(self, x):
#         """
#         Args:
#             x: (B, embedding_dims)
#         """
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x) # (B, 3 * pcd_samples)
#         # x = x.view(-1, 3, self.pcd_samples) # (B, 3, pcd_samples)
#         x = x.view(-1, self.pcd_samples, 3) # (B, 3, pcd_samples)
#         return x


# # class PCDEncoder(nn.Module):
# #     def __init__(self, embedding_dims=1024):
# #         super(PCDEncoder, self).__init__()
# #         self.embedding_dims = embedding_dims
# #         self.conv1 = nn.Conv1d(3, 256, 1)
# #         self.conv2 = nn.Conv1d(256, 512, 1)
# #         self.conv3 = nn.Conv1d(512, embedding_dims, 1)

# #     def forward(self, x):
# #         x = F.relu(self.conv1(x))
# #         x = F.relu(self.conv2(x))
# #         x = self.conv3(x)
# #         x = torch.max(x, 2, keepdim=True)[0]
# #         x = x.view(-1, self.embedding_dims)
# #         return x

# # class PCDDecoder(nn.Module):
# #     def __init__(self, embedding_dims=1024, pcd_samples=2048):
# #         super(PCDDecoder, self).__init__()
# #         self.fc1 = nn.Linear(embedding_dims, 512)
# #         self.fc2 = nn.Linear(512, 1024)
# #         self.fc3 = nn.Linear(1024, pcd_samples*3)
# #         self.pcd_samples = pcd_samples

# #     def forward(self, x):
# #         x = F.relu(self.fc1(x))
# #         x = F.relu(self.fc2(x))
# #         x = self.fc3(x)
# #         x = x.view(-1, 3, self.pcd_samples)
# #         return x
