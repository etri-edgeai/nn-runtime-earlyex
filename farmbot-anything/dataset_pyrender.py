import os 
import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch3d as p3d
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
    TexturesUV
)


# Load config file
cfg_path = "./config/base.yml"
with open(cfg_path,'r') as f:
    cfg = yaml.safe_load(f)

# Load metadata
metadata_df = pd.read_csv(cfg['shapenet']['metadata'])
device = torch.device("cuda:0")
# Choose an item from the metadata
index = 10
item = metadata_df.iloc[index]
img_id = item['fullId']
category = item['category']
print(item)

# Set paths for obj, mtl and texture files
obj_path = os.path.join(cfg['shapenet']['obj'], img_id[4:]) + '.obj'

# obj_path = "./tomato.obj"
# Load the mesh
mesh = p3d.io.load_objs_as_meshes([obj_path], device="cuda")

if isinstance(mesh.textures, TexturesUV):
    print("The mesh has texture.")
else:
    print("The mesh does not have texture.")
    verts, faces, _ = load_obj(obj_path)

    # Create a Mesh object from the verts and faces tensors
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx]).to(device)
    faces_tensor = mesh.faces_packed().to(torch.long)
    valid_faces = faces_tensor[faces_tensor.gt(-1).all(1)]

    texture_image = torch.ones((1, verts.shape[0], 3),dtype=torch.float32).to(device)
    texture = TexturesVertex(verts_features=texture_image)
    mesh.textures = texture


# Normalize the mesh
verts = mesh.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)
mesh.scale_verts_((1.0 / float(scale)))

# Define number of views
num_views = 10

# Define camera positions and orientations
elev = torch.rand(num_views) * 180  # Random elevation angles between 0 and 180 degrees
azim = torch.rand(num_views) * 360 - 180  # Random azimuth angles between -180 and 180 degrees
distances = torch.rand(num_views) * 18 + 1  # Random distances between 2 and 20 units

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

R, T = look_at_view_transform(dist=distances, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...]).to(device)
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]]).to(device)

raster_settings = RasterizationSettings(
    image_size=1024, 
    blur_radius=0.0, 
    faces_per_pixel=4,
    bin_size=0
)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera,
        lights=lights
    )
)

# Render the mesh from each viewing angle
meshes = mesh.extend(num_views)
target_images = renderer(meshes, cameras=cameras, lights=lights)
target_rgb = [target_images[i, ..., :3] for i in range(num_views)]

# Save the rendered images to disk
for i in range(num_views):
    plt.imsave(f"rendered_{i}.png", target_rgb[i].cpu().numpy())

    # Print the 6D pose for each view
    R_i = R[i].cpu().numpy()
    T_i = T[i].cpu().numpy()
    pose = np.eye(4)
    pose[:3, :3] = R_i
    pose[:3, 3] = T_i
    print(f"View {i}:")
    print("6D pose:")
    print(pose)
    print()