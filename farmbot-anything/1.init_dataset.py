import json
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch3d as p3d
import torch
import torchvision
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import (CamerasBase, FoVPerspectiveCameras,
                                HardFlatShader, HardPhongShader,
                                MeshRasterizer, MeshRenderer, PointLights,
                                RasterizationSettings, SoftPhongShader,
                                SoftSilhouetteShader, Textures, TexturesAtlas,
                                TexturesUV, look_at_view_transform, 
                                PointsRenderer, PointsRasterizer, 
                                PointsRasterizationSettings, 
                                NormWeightedCompositor, AlphaCompositor)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from tqdm import tqdm

import lmdb


def color_points_white(point_cloud):
    # Create an array of the same shape as the point cloud with all values set to 1
    colors = torch.ones(point_cloud.shape[0], 3, device=point_cloud.device)
    return colors

class DepthShader(SoftPhongShader):
    def __init__(self, device, cameras=None, lights=None, blend_params=None):
        if lights is None:
            lights = PointLights(device=device, ambient_color=((1,1,1)))

        super().__init__(device=device, cameras=cameras, lights=lights)

    def transform(self, fragments, meshes, **kwargs):
        # Get the depth from the z-buffer of the fragments
        depth = fragments.zbuf[..., :1]
        # Normalize or scale the depth as needed
        # For simplicity, we'll just return it as is
        return depth

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16,np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): 
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump(
            {'info': info, 'licenses': licenses, 'images': images, 
                'annotations': annotations, 'categories': categories}, 
            coco, indent=2, sort_keys=True)

def save_mask_and_bbox(seg_mask, bbox, output_path="output.png"):
    # Convert the PyTorch tensor to a NumPy array.
    # Assuming seg_mask is a 2D tensor (height x width) of dtype torch.bool.
    seg_mask_np = seg_mask.cpu().numpy().astype(np.uint8) * 255  # Convert boolean mask to uint8 {0, 255} mask.

    # Create a 3-channel version of the segmentation mask.
    seg_mask_three_channel = cv2.merge([seg_mask_np, seg_mask_np, seg_mask_np])

    # Draw the bounding box on the image.
    # Note: OpenCV's rectangle function expects the coordinates in (x, y) format.
    color = (0, 255, 0)  # Green color in BGR.
    thickness = 2  # Line thickness for the bounding box.
    start_point = (bbox[0], bbox[1])  # min_x, min_y.
    end_point = (bbox[2], bbox[3])  # max_x, max_y.
    image_with_bbox = cv2.rectangle(seg_mask_three_channel, start_point, end_point, color, thickness)

    # Save the image with the bounding box.
    cv2.imwrite(output_path, image_with_bbox)

def create_lmdb_from_df(df, cfg, lmdb_path, map_size=1e12):
    device          = torch.device("cuda:0")
    num_views       = cfg['1_num_views']
    img_size        = cfg['0_img_size']
    pcd_samples     = cfg['0_pcd_num']
    a0, a1          = cfg['1_azimuth_range_0'], cfg['1_azimuth_range_1']
    e0, e1          = cfg['1_elevation_range_0'], cfg['1_elevation_range_1']
    d0, d1          = cfg['1_distance_range_0'], cfg['1_distance_range_1']
    env = lmdb.open(lmdb_path, map_size=1*512*1024*1024*1024)

    with env.begin(write=True) as txn:
        pbar = tqdm(range(len(df)))
        num_count = 0
        for index in pbar:
            item = df.iloc[index]
            img_id = item['fullId']
            indexnum = item['old_index']
            up_vector = item['up']
            front_vector = item['front']
            surfaceVolume = item['surfaceVolume']
            weight = item['weight']
            brix = item['brix']
            red_green = item['red/green']

            obj_path = os.path.join(cfg['1_obj_dir'], str(indexnum))
            print(img_id)
            obj_path = os.path.join(obj_path, img_id)+'.obj'
            category_id = int(item['category_id'])

            pbar.set_description(f"Loading {obj_path}")
            mesh = p3d.io.load_objs_as_meshes([obj_path], device=device)
            if isinstance(mesh.textures, TexturesUV):
                mesh
            else:
                verts, faces, aux = load_obj(
                    obj_path, load_textures=True, 
                    create_texture_atlas=True, texture_atlas_size=4)
                mesh = Meshes(
                    verts=[verts], faces=[faces.verts_idx], 
                    textures=TexturesAtlas(atlas=[aux.texture_atlas])
                    ).to(device)
            # Normalize the mesh
            verts = mesh.verts_packed()
            center = verts.mean(0)
            # scale = max((verts - center).abs().max(0)[0])
            scale = (verts - center).abs().max() 
            mesh.offset_verts_(-center)
            mesh.scale_verts_((1.0 / float(scale)))
        
            for rand in range(cfg['1_num_rand']):
                # Define distance as a linspace from 2 to 6
                distances = torch.linspace(d0, d1, num_views)
                # this?
                # azim = torch.arange(a0, a1, (a1 - a0) / num_views)
                # elev = torch.arange(e0, e1, (e1 - e0) / num_views)
                # or this?
                azim = torch.rand(num_views) * (a1 - a0) + a0
                elev = torch.rand(num_views) * (e1 - e0) + e0          

                # Create a batch of meshes by repeating the current mesh
                meshes = mesh.extend(num_views)

                # Get the bounding box
                bbox_3d = mesh.get_bounding_boxes()
                
                # Sample points from the surface of the mesh
                points = sample_points_from_meshes(meshes=meshes, num_samples=50000)
                points = sample_farthest_points(points, K=pcd_samples)[0]

                # Create multiple camera views
                pbar.set_description(f"[{num_count}]Creating cameras...")
                R, T = look_at_view_transform(
                    dist=distances, elev=elev, azim= azim)

                camera = FoVPerspectiveCameras(
                    device=device, R=R, T=T).to(device)

                lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

                # Create a RasterizationSettings object
                blend_params = BlendParams(1e-4, 1e-4, (0,0,0))
                
                # Create a MeshRasterizer object for the rgb image
                pbar.set_description(f"[{num_count}]Rendering rgb image...")
                rgb_settings = RasterizationSettings(
                    image_size=img_size, 
                    blur_radius=0.0, 
                    faces_per_pixel=4, 
                    bin_size=0)
                rgb_rasterizer = MeshRasterizer(
                    cameras=camera, raster_settings=rgb_settings)
                rgb_shader = SoftPhongShader(
                    device=device, cameras=camera, 
                    lights=lights, blend_params=blend_params)
                rgb_renderer = MeshRenderer(
                    rasterizer=rgb_rasterizer, shader=rgb_shader)
                rgb_images = rgb_renderer(meshes, cameras=camera, lights=lights)
                rgb_targets = [rgb_images[i, ..., :3] for i in range(num_views)]

                # Create a MeshRasterizer object for the segmentation image
                pbar.set_description(f"[{num_count}]Rendering segmentation image...")
                seg_raster_settings = RasterizationSettings(
                    image_size=img_size, blur_radius=0.0, 
                    faces_per_pixel=1, bin_size=0)
                seg_rasterizer = MeshRasterizer(
                    cameras=camera, raster_settings=seg_raster_settings)
                seg_shader = HardFlatShader(
                    device=device, cameras=camera, 
                    lights=lights, blend_params=blend_params)
                seg_renderer = MeshRenderer(
                    rasterizer=seg_rasterizer, shader=seg_shader)
                seg_images = seg_renderer(meshes, cameras=camera)
                seg_masks = [(seg_images[i, ..., 3] > 0) for i in range(num_views)]

                # Create a rasterizer for the depth image
                pbar.set_description(f"[{num_count}]Rendering depth image...")
                depth_raster_settings = RasterizationSettings(
                    image_size=img_size, blur_radius=0.0, 
                    faces_per_pixel=1, bin_size=0)
                depth_raster_settings.perspective_correct = True
                depth_rasterizer = MeshRasterizer(
                    cameras=camera, raster_settings=depth_raster_settings)
                depth_fragments = depth_rasterizer(meshes, camera=camera)
                depth_zbuf = depth_fragments.zbuf
                depth_images = depth_zbuf.min(dim=-1).values
                normalized_depth = (depth_images - depth_images.min()) / (
                    depth_images.max() - depth_images.min())
                depth_image_8bit = (255 * normalized_depth).to(dtype=torch.uint8)

                # Create a rasterizer for the point cloud
                pbar.set_description(f"[{num_count}]Rendering point cloud...")
                pcd_raster_settings = PointsRasterizationSettings(
                    image_size=img_size, radius=0.01, points_per_pixel=10)
                pcd_rasterizer = PointsRasterizer(
                    cameras=camera, raster_settings=pcd_raster_settings)
                pcd_renderer = PointsRenderer(
                    rasterizer=pcd_rasterizer, compositor=AlphaCompositor())

                # print(points.shape)
                pcdd = points.squeeze()
                # print(pcdd.shape)
                # colors = color_points_white(points)
                colors = color_points_white(points).unsqueeze(1).expand(-1, pcdd.shape[1], -1)

                # print(colors.shape)
                pcd_point_clouds = Pointclouds(points=pcdd, features=colors)
                # print(pcd_point_clouds.points_padded().shape)
                pcd_images = pcd_renderer(pcd_point_clouds)
                # print(pcd_images.shape)
                pcd_targets = [pcd_images[0,..., :3] for i in range(num_views)]

                # Save the rendered images to disk
                pbar.set_description(f"[{num_count}]Saving images to disk...")

                for view in range(num_views):
                    idx = index * cfg['1_num_rand'] * num_views + rand * num_views + view
                    R_i, T_i = R[view].to(device), T[view].to(device)
                    pose = torch.eye(4, device="cpu")
                    pose[:3, :3] = R[view].cpu()
                    pose[:3,  3] = T[view].cpu()
                    
                    seg_mask = seg_masks[view]                   
                    nonzero_pixels = torch.nonzero(seg_mask)

                    if nonzero_pixels.numel() > 0:
                        # Create a unique key for each image
                        key_prefix = f"{num_count}_"

                        # Put RGB Image
                        txn.put((key_prefix+'image').encode('utf-8'), rgb_targets[view].float().cpu().numpy().tobytes())
                        rgb_pil = (rgb_targets[view] * 255).to(torch.uint8)
                        rgb_pil = Image.fromarray(rgb_pil.cpu().numpy())
                        rgb_pil.save(os.path.join("temp", f"{num_count}_rgb.png"))

                        # Put Segmentation Mask
                        txn.put((key_prefix+'mask').encode('utf-8'), seg_masks[view].cpu().numpy().tobytes())

                        # Put Depth Image
                        normalized_depth = (depth_images - depth_images.min()) / (depth_images.max() - depth_images.min())
                        depth_image_8bit = (normalized_depth * 255).to(torch.uint8)
                        depth_pil = Image.fromarray(depth_image_8bit[view].cpu().numpy())
                        depth_pil.save(os.path.join("temp", f"{num_count}_depth.png"))      
                        txn.put((key_prefix+'depth').encode('utf-8'), normalized_depth[view].cpu().numpy().tobytes())

                        # Put Bounding Box
                        min_x = nonzero_pixels[:, 1].min().item()
                        max_x = nonzero_pixels[:, 1].max().item()
                        min_y = nonzero_pixels[:, 0].min().item()
                        max_y = nonzero_pixels[:, 0].max().item()
                        bbox = np.array([min_x, min_y, max_x, max_y]).astype(np.int32)
                        txn.put((key_prefix+'bbox').encode('utf-8'), bbox.tobytes())
                       
                        # Put Point Cloud
                        txn.put((key_prefix+'rpcd').encode('utf-8'), pcd_targets[view].cpu().numpy().tobytes())
                        txn.put((key_prefix+'pcd').encode('utf-8'), points[view].cpu().numpy().tobytes())
 
                        # Put Pose
                        txn.put((key_prefix+'pose').encode('utf-8'), pose.cpu().numpy().tobytes())

                        # Put Category ID
                        txn.put((key_prefix+'category_id').encode('utf-8'), np.array([category_id]).tobytes())

                        # Put Up Vector, Front Vector
                        up = [float(i) for i in up_vector.split('\\,')] 
                        front = [float(i) for i in front_vector.split('\\,')]
                        txn.put((key_prefix+'up_vector').encode('utf-8'), np.array(up, dtype=np.float32).tobytes())
                        txn.put((key_prefix+'front_vector').encode('utf-8'), np.array(front, dtype=np.float32).tobytes())
                        
                        # Put Surface Volume, Weight, Brix, Ripeness
                        txn.put((key_prefix+'surfaceVolume').encode('utf-8'), np.array(surfaceVolume, dtype=np.float32).tobytes())
                        txn.put((key_prefix+'weight').encode('utf-8'), np.array(weight, dtype=np.float32).tobytes())
                        txn.put((key_prefix+'brix').encode('utf-8'), np.array(brix, dtype=np.float32).tobytes())
                        txn.put((key_prefix+'ripeness').encode('utf-8'), np.array(red_green, dtype=np.float32).tobytes())
                        
                        # Put Index
                        num_count += 1
                    else:
                        print("No nonzero pixels found")
                        bbox = np.array([0, 0, 0, 0]).astype(np.int32)

                        output_path = os.path.join("output", f"{num_count}_seg_mask.png")
                        save_mask_and_bbox(seg_mask, bbox, output_path)
                        rgb_output_path = os.path.join("output", f"{num_count}_rgb.png")
                        plt.imsave(rgb_output_path, rgb_targets[view].cpu().numpy())
        
        # Put the number of images
        txn.put((b"__len__"), str(num_count).encode('utf-8'))
def main(cfg):
    '''
    main function
    args:
        cfg: config file
    '''
    print("Fruit Ripeness & Utility Measure Tool(FRUT)")
    print("Starting dataset_single_test.py...")

    # # Read the config file
    output_dir  = cfg['1_dataset_dir']
    print(f"Loading metadata...{cfg['1_input_csv']}")
    df = pd.read_csv(cfg['1_input_csv'])

    # # Choose an item from the metadata
    print("splitting object...")
    value_counts = df['category_id'].value_counts()
    classes_to_keep = value_counts[value_counts > 1].index
    filtered_df = df[df['category_id'].isin(classes_to_keep)]
    train_df, test_df = train_test_split(
        filtered_df, test_size=0.1, stratify=filtered_df['category_id'], random_state=42)
    train_df.to_csv(os.path.join(output_dir,'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir,'test.csv'), index=False)

    # Create Train HDF5
    print("Creating Train LMDB...")
    create_lmdb_from_df(train_df, cfg, cfg['1_lmdb_train'])

    print("Creating Test LMDB...")
    create_lmdb_from_df(test_df, cfg, cfg['1_lmdb_test'])


    # filtered_df.to_csv(os.path.join(output_dir,'train.csv'), index=False)
    # create_lmdb_from_df(filtered_df, cfg, cfg['1_lmdb_train'])

if __name__ == "__main__":
    device = torch.device("cuda:3")
    cfg_path = "./config/config.yml"
    with open(cfg_path,'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)



