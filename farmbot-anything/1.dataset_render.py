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
from pytorch3d.ops import sample_points_from_meshes
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

def main(cfg):
    '''
    main function
    args:
        cfg: config file
    '''
    print("Starting dataset_single_test.py...")

    # Read the config file
    img_size    = cfg['0_img_size']
    pcd_samples = cfg['0_pcd_num']
    num_views   = cfg['1_num_views']

    d_0 = cfg['1_distance_range_0']
    d_1 = cfg['1_distance_range_1']
    a_0 = cfg['1_azimuth_range_0']
    a_1 = cfg['1_azimuth_range_1']
    e_0 = cfg['1_elevation_range_0']
    e_1 = cfg['1_elevation_range_1']

    output_dir = cfg["1_dataset_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading metadata...{cfg['1_input_csv']}")
    df = pd.read_csv(cfg['1_input_csv'])
    obj_dir = cfg['1_obj_dir']

    # Create output directories
    rgb_dir = os.path.join(output_dir, 'rgb')
    json_dir = os.path.join(output_dir, 'json')
    depth_dir = os.path.join(output_dir, 'depth')
    seg_dir = os.path.join(output_dir, 'seg')
    pcd_dir = os.path.join(output_dir, 'pcd')
    apcd_dir = os.path.join(output_dir, 'apcd')
    rpcd_dir = os.path.join(output_dir, 'rpcd')
    pose_dir = os.path.join(output_dir, 'pose')
    
    print("Creating output directories...")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)
    os.makedirs(apcd_dir, exist_ok=True)
    os.makedirs(rpcd_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    # # Choose an item from the metadata
    total_length = len(df)
    print("Total length: ", total_length)
    pbar = tqdm(range(total_length))
    img_num = 0

    print("Creating categories...")
    coco_output = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    print("Creating categories...")
    classes = df['category'].unique()
    for i, cls in enumerate(classes):
        coco_output["categories"].append({
            'id': i + 1,
            'name': cls,
            'supercategory': 'object',
        })

    print("Creating images...")
    try: 
        for index in pbar:
            # Get the item
            item = df.iloc[index]
            img_id = item['fullId'][4:]
            category = item['category']
            category_id = item['category_id'] +1
            obj_path = os.path.join(obj_dir, img_id) + '.obj'
            up = torch.tensor([
                float(x) for x in item['up'].split('\\,')], device=device)
            front = torch.tensor([
                float(x) for x in item['front'].split('\\,')], device=device)

            # Define distance as a linspace from 2 to 6
            distances = torch.linspace(d_0, d_1, num_views)
            azim = torch.arange(a_0, a_1, (a_1-a_0) / num_views)
            elev = torch.arange(e_0, e_1, (e_1-e_0) / num_views)

            # Load obj file
            pbar.set_description("Loading obj file...")
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
            scale = max((verts - center).abs().max(0)[0])
            mesh.offset_verts_(-center)
            mesh.scale_verts_((1.0 / float(scale)))
            
            # Create a batch of meshes by repeating the current mesh
            meshes = mesh.extend(num_views)

            # Get the bounding box
            bbox_3d = mesh.get_bounding_boxes()
            
            # Sample points from the surface of the mesh
            points = sample_points_from_meshes(meshes=meshes, num_samples=pcd_samples)

            # Create multiple camera views
            pbar.set_description("Creating cameras...")
            R, T = look_at_view_transform(
                dist=distances, elev=elev, azim= azim)

            camera = FoVPerspectiveCameras(
                device=device, R=R, T=T).to(device)

            # Create a PointLights object
            lights = PointLights(
                device=device, location=[[0.0, 0.0, -2.0]]).to(device)
            
            # Create a RasterizationSettings object
            blend_params = BlendParams(1e-4, 1e-4, (0,0,0))
            
            # Create a MeshRasterizer object for the rgb image
            pbar.set_description("Rendering rgb image...")
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
            pbar.set_description("Rendering segmentation image...")
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
            pbar.set_description("Rendering depth image...")
            depth_raster_settings = RasterizationSettings(
                image_size=img_size, blur_radius=0.0, 
                faces_per_pixel=1, bin_size=0)
            depth_raster_settings.perspective_correct = True
            depth_rasterizer = MeshRasterizer(
                cameras=camera, raster_settings=depth_raster_settings)
            depth_fragments = depth_rasterizer(meshes, camera=camera)
            depth_zbuf = depth_fragments.zbuf
            depth_images = depth_zbuf.min(dim=-1).values

            # Create a rasterizer for the point cloud
            pbar.set_description("Rendering point cloud...")
            
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
            pbar.set_description("Saving images to disk...")
            for i in range(num_views):
                # Save the rendered RGB images to disk
                rgb_path = f"{rgb_dir}/{img_id}_{i}.png"
                # plt.imshow(rgb_targets[i].cpu().numpy())
                # plt.axis('off')
                # plt.text(10, 10, f"View {rgb_path}", color="white")       
                # plt.savefig(rgb_path, bbox_inches='tight', pad_inches=0)
                # plt.close()
                plt.imsave(rgb_path, rgb_targets[i].cpu().numpy())

                # Save the segmentation masks to disk
                seg_path = f"{seg_dir}/{img_id}_{i}.png"
                seg_mask = seg_masks[i].detach().cpu().numpy()
                Image.fromarray(seg_mask).save(seg_path)

                # Save the rendered point cloud to disk
                rpcd_path = f"{rpcd_dir}/{img_id}_{i}.png"
                rpcd_i = pcd_images[i].detach().cpu().numpy()
                rpcd_image = (rpcd_i - rpcd_i.min()) / (rpcd_i.max() - rpcd_i.min())
                plt.imsave(rpcd_path, rpcd_image)

                # Rotation and translation
                R_i, T_i = R[i].to(device), T[i].to(device)

                # Save Point Cloud with Absolute Coordinates to disk
                apcd_path = f"{apcd_dir}/{img_id}_{i}.pt"
                apcd_i = points[i].unsqueeze(0).to(device)
                torch.save(apcd_i, apcd_path)

                # Save Point Cloud with Viewpoint Coordinates to disk
                pcd_path = f"{pcd_dir}/{img_id}_{i}.pt"
                vpcd_i = torch.matmul(apcd_i - T_i , R_i)
                vpcd_i -= vpcd_i.mean(dim=0)
                # Scale the point cloud
                scale = vpcd_i.abs().max()
                vpcd_i /= scale
                torch.save(vpcd_i, pcd_path)


                # Save the depth images to disk
                depth_path = f"{depth_dir}/{img_id}_{i}.pt"
                torch.save(depth_images[i], depth_path)

                # Save the pose to disk
                pose_path = f"{pose_dir}/{img_id}_{i}.pt"
                pose = torch.eye(4, device="cpu")
                pose[:3, :3] = R[i].cpu()
                pose[:3,  3] = T[i].cpu()
                torch.save(pose, pose_path)


                # Save the image to the coco format
                coco_output["images"].append({
                    'id': img_num,
                    'width': img_size,
                    'height': img_size,
                    'rgb_path': rgb_path,
                    'seg_path' : seg_path,
                    'pcd_path' : pcd_path,
                    'apcd_path' : apcd_path,
                    'depth_path' : depth_path,
                    'rpcd_path' : rpcd_path,
                    'pose_path' : pose_path,
                    'license': 0,
                    'flickr_url': '',
                    'coco_url': '',
                    'date_captured': '',
                })
                


                # save the 6D pose to disk
                bbox_3d_i   = np.round(bbox_3d.cpu().numpy(), 1)
                distance_i  = np.round(distances[i].cpu().numpy(), 1)
                elev_i      = np.round(elev[i].cpu().numpy(), 1)
                azim_i      = np.round(azim[i].cpu().numpy(), 1)
                scale_i     = np.round(scale.cpu().numpy(), 1)
                center_i    = np.round(center.cpu().numpy(), 1)

                # Create a bounding box
                seg_img = Image.open(seg_path).convert('L')
                seg_img = torchvision.transforms.ToTensor()(seg_img).to(device)
                seg_img = cv2.imread(seg_path,cv2.IMREAD_GRAYSCALE)
                _, binary_img = cv2.threshold(seg_img, 0, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                    binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = [x, y, w, h]

                # Save up and front vectors
                up = torch.tensor([
                    float(x) for x in item['up'].split('\\,')], 
                    device=device)[None, None, :]
                front = torch.tensor([
                    float(x) for x in item['front'].split('\\,')], 
                    device=device)[None, None, :]
                
                # Save up and front vectors in camera coordinates
                up_cam_i = torch.matmul(
                    R[i][None, ...], 
                    up.cpu().unsqueeze(-1)).squeeze(-1) + T[i][None, ...]
                front_cam_i = torch.matmul(
                    R[i][None, ...], 
                    front.cpu().unsqueeze(-1)).squeeze(-1) + T[i][None, ...]

                # COCO format
                coco_output["annotations"].append({
                    'R':            R_i.tolist(),
                    'T':            T_i.tolist(),
                    'area':         0,
                    'bbox':         bbox,
                    'id':           img_num,
                    'scale':        scale_i.tolist(),
                    'center':       center_i.tolist(),
                    'image_id':     img_num,
                    'category_id':  category_id,
                    'up_vector':    up_cam_i.tolist(),
                    'front_vector': front_cam_i.tolist(),
                    '3d_bbox':      bbox_3d_i.tolist(),
                    'azimuth':      azim_i.tolist(),
                    'seg_path' :    seg_path,
                    'pcd_path' :    pcd_path,
                    'apcd_path' :   apcd_path,
                    'depth_path' :  depth_path,
                    'distance':     distance_i.tolist(),
                    'elevation':    elev_i.tolist(),
                    'segmentation': [],
                    'iscrowd': 0,
                })

                # increment image number
                img_num += 1

    except KeyboardInterrupt:
        pass
    finally:
        with open(cfg['1_dataset_json'], "w") as f:
            json.dump(coco_output, f,cls=NumpyEncoder)

    # Split the dataset into train and test
    print("Splitting dataset into train and test...")
    with open(cfg['1_dataset_json'], 'r') as f:
        coco = json.load(f)
        print(len(coco['images']))
        info = coco['info']
        images = coco['images']
        licenses = coco['licenses']
        annotations = coco['annotations']
        categories = coco['categories']

        # Split the dataset into train and test
        train, test = train_test_split(images, test_size=0.2)
        train_path = cfg['1_train_json']
        test_path = cfg['1_test_json']

        # save coco format
        save_coco(train_path, info, licenses, train, annotations, categories)
        save_coco(test_path, info, licenses, test, annotations, categories)
        print(len(train), len(test))

if __name__ == "__main__":
    device = torch.device("cuda")
    cfg_path = "./config/config.yml"
    with open(cfg_path,'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
