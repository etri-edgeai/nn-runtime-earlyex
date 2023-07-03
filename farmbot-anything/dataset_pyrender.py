import os 
import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch3d as p3d
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
import cv2
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesAtlas,
    HardFlatShader,
    HardPhongShader,
    CamerasBase,
    SoftSilhouetteShader
)
from tqdm import tqdm

from PIL import Image
import json
# from pytorch3d.ops import get_bounding_boxes
from pytorch3d.renderer.blending import BlendParams

num_views = 16
img_size = 1024
device = torch.device("cuda:1")
cfg_path = "./config/shapenet.yml"
with open(cfg_path,'r') as f:
    cfg = yaml.safe_load(f)

# Load metadata
df = pd.read_csv(cfg['shapenet']['metadata4'])
output_dir = cfg['shapenet']['rendered']

# Choose an item from the metadata
total_length = len(df)
print("Total length: ", total_length)

class DepthShader(SoftPhongShader):
    def __init__(self, device, cameras=None, lights=None, blend_params=None):
        if lights is None:
            lights = PointLights(device=device, ambient_color=((1,1,1),))

        super().__init__(device=device, cameras=cameras, lights=lights)

    def transform(self, fragments, meshes, **kwargs):
        # Get the depth from the z-buffer of the fragments
        depth = fragments.zbuf[..., :1]
        # Normalize or scale the depth as needed
        # For simplicity, we'll just return it as is
        return depth



for index in tqdm(range(total_length)):
    # index = 130
    item = df.iloc[index]
    img_id = item['fullId'][4:]
    category = item['category']
    # print(item)

    # Set paths for obj, mtl and texture files
    obj_path = os.path.join(cfg['shapenet']['obj'], img_id) + '.obj'

    mesh = p3d.io.load_objs_as_meshes([obj_path], device=device)
    if isinstance(mesh.textures, TexturesUV):
        mesh
    else:
        # print("The mesh does not have texture.")
        verts, faces, aux = load_obj(obj_path, load_textures=True, create_texture_atlas=True, texture_atlas_size=4)
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures= TexturesAtlas(atlas=[aux.texture_atlas])).to(device)
        
    # Normalize the mesh
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    # Define distance as a linspace from 2 to 6
    distances = torch.linspace(2, 6, num_views)

    # Define azimuth and elevation to rotate and cover all angles
    azim = torch.linspace(20, 340, num_views)  # Covers full circle around the object
    elev = torch.linspace(20, 160, num_views)  # Varied elevations to cover from top to bottom views

    R, T = look_at_view_transform(dist=distances, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...]).to(device)
    lights = PointLights(device=device, location=[[0.0, 0.0, -5.0]]).to(device)

    raster_settings = RasterizationSettings(
        image_size=img_size, 
        blur_radius=0.0, 
        faces_per_pixel=4,
        bin_size=0)

    # Background image color
    blend_params = BlendParams(1e-4, 1e-4, (0,0,0))

    # Create a MeshRasterizer object for the RGB image
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=camera, lights=lights,
            blend_params=blend_params
        ))

    # Get the bounding boxes for the mesh
    bbox_3d = mesh.get_bounding_boxes()

    # Render the mesh from each viewing angle
    meshes = mesh.extend(num_views)
    target_images = renderer(meshes, cameras=cameras, lights=lights)
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]

    # Create a MeshRasterizer object for the segmentation image
    seg_raster_settings = RasterizationSettings(
        image_size=img_size, 
        blur_radius=0.0, 
        faces_per_pixel=2,
        bin_size=0)

    seg_rasterizer = MeshRasterizer(
        cameras=camera, 
        raster_settings=seg_raster_settings
    )
    # Create a renderer for the segmentation image
    seg_renderer = MeshRenderer(
        rasterizer=seg_rasterizer,
        shader=HardFlatShader(
            device=device,
            cameras=camera,
            lights=lights,
            blend_params=blend_params
        )
    )
    seg_images = seg_renderer(meshes, cameras=cameras)
    seg_masks = [(seg_images[i, ..., 3] > 0) for i in range(num_views)]

    # Create a rasterizer for the depth image
    depth_raster_settings = RasterizationSettings(
        image_size=img_size, 
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0)
    depth_raster_settings.perspective_correct = True

    # Create a MeshRasterizer object for the depth image
    depth_rasterizer = MeshRasterizer(
        cameras=camera,
        raster_settings=depth_raster_settings)

    fragments = depth_rasterizer(meshes, camera=cameras)
    zbuf = fragments.zbuf
    depth_images = zbuf.min(dim=-1).values

    # Save the rendered images to disk
    for i in range(num_views):
        # Save the rendered RGB images to disk
        rend_path = f"{output_dir}rendered/{img_id}_{i}.png"
        plt.imsave(rend_path, target_rgb[i].cpu().numpy())
        # print(rend_path) 

        # Save the segmentation masks to disk
        seg_path = f"{output_dir}segment/{img_id}_{i}.png"
        seg_mask = seg_masks[i].detach().cpu().numpy()
        Image.fromarray(seg_mask).save(seg_path)
        # seg_mask_norm = (seg_masks[i] - seg_masks[i].min()) / (seg_masks[i].max() - seg_masks[i].min())

        # plt.imsave(seg_path, seg_mask_norm.cpu().numpy())


        # print(seg_path)
        depth_path = f"{output_dir}depth/{img_id}_{i}.npy"
        depth_npy = depth_images[i].detach().cpu().numpy()
        np.save(depth_path, depth_npy)

        # Print the 6D pose for each view
        R_i = R[i].cpu().numpy()
        T_i = T[i].cpu().numpy()
        pose = np.eye(4)
        pose[:3, :3] = R_i
        pose[:3, 3] = T_i

        bbox_3d_i = np.round(bbox_3d.cpu().numpy(), 1)
        distance_i = np.round(distances[i].cpu().numpy(), 1)
        elev_i = np.round(elev[i].cpu().numpy(), 1)
        azim_i = np.round(azim[i].cpu().numpy(), 1)
        scale_i = np.round(scale.cpu().numpy(), 1)
        center_i = np.round(center.cpu().numpy(), 1)


        seg_img = cv2.imread(seg_path,cv2.IMREAD_GRAYSCALE)
        _, binary_img = cv2.threshold(seg_img, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bbox_2d = [x, y, x+w, y+h]

        up = torch.tensor([float(x) for x in item['up'].split('\\,')], device=device)
        front = torch.tensor([float(x) for x in item['front'].split('\\,')], device=device)

        up = up[None, None, :]
        front = front[None, None, :]
        # Apply the rotation and translation to the 'up' and 'front' vectors for the i-th view
        # print(R[i].device, up.device,front.device, T[i].device)
        up_cam_i = torch.matmul(R[i][None, ...], up.cpu().unsqueeze(-1)).squeeze(-1) + T[i][None, ...]
        front_cam_i = torch.matmul(R[i][None, ...], front.cpu().unsqueeze(-1)).squeeze(-1) + T[i][None, ...]


        data = {
            'img_id': img_id,
            'view_id': i,
            'category': category,
            'cam_pose': pose.tolist(),
            'up_vector': up_cam_i.tolist(),
            'front_vector': front_cam_i.tolist(),
            'unit': item['unit'],
            'surfaceVolume': item['surfaceVolume'],
            'solidVolume': item['solidVolume'],
            'weight': item['weight'],
            '3d_bbox': bbox_3d_i.tolist(),
            '2d_bbox': bbox_2d,
            'distance': distance_i.tolist(),
            'elevation': elev_i.tolist(),
            'azimuth': azim_i.tolist(),
            'scale': scale_i.tolist(),
            'center': center_i.tolist()
        }
        # Save the 6D pose to disk
        json_path = f"{output_dir}pose/{img_id}_{i}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)
        

        # print(f"View {i}:")
        # print("original up Vector:")
        # print("6D pose:")
        # # print(np.round(pose,1))
        # print("Bounding box:")
        # print(np.round(bbox_3d.cpu().numpy(),1))
        # print("Distance:")
        # print(np.round(distances[i].cpu().numpy(),1))
        # print("Elevation:")
        # print(np.round(elev[i].cpu().numpy(),1))
        # print("Azimuth:")
        # print(np.round(azim[i].cpu().numpy(),1))
        # print("Scale:")
        # print(np.round(scale.cpu().numpy(),1))
        # print("Center:")
        # print(np.round(center.cpu().numpy(),1))