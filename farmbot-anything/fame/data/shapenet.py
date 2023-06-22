import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

class ShapeNetDataset(Dataset):
    def __init__(self, root_dir, categories, transform=None):
        self.root_dir = root_dir
        self.categories = categories
        self.transform = transform
        self.mesh_list = []

        for category in self.categories:
            category_dir = os.path.join(self.root_dir, category)
            mesh_list = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.obj')]
            self.mesh_list += mesh_list

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        mesh_path = self.mesh_list[idx]
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)

        if self.transform:
            verts = self.transform(verts)

        return verts, faces

# Set the path to the root directory of the ShapeNetCore dataset
root_dir = "/path/to/ShapeNetCore.v2"

# Define the list of categories to include in the dataset
categories = ['chair', 'table']

# Define any data transforms to be applied to the input meshes
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset and dataloader
dataset = ShapeNetDataset(root_dir, categories, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)