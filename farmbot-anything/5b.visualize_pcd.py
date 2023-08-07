import json
import os
import sys
from argparse import ArgumentParser
import time
import torch
import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
import cv2
import shutil
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance
from torch.autograd import Variable
# from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
# import horovod.torch as hvd

cwd = os.getcwd()
sys.path.append(cwd)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
from modules.data.srendered import ShapeNetRenderedDataset, custom_collate_fn
from modules.se3rf import E3RFnet
from torch.utils.data.distributed import DistributedSampler
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

def get_distances(point_cloud, reference_point):
    # Calculate the distance from the reference point for each point in the point cloud
    distances = torch.norm(point_cloud - reference_point[None, :], dim=-1)
    return distances

def color_points_white(point_cloud):
    # Create an array of the same shape as the point cloud with all values set to 1
    colors = torch.ones(point_cloud.shape[0], 3, device=point_cloud.device)
    return colors

def color_points(distances):
    # Normalize the distances
    distances = (distances - distances.min()) / (distances.max() - distances.min())
    # Create a colormap
    colormap = plt.get_cmap("cool")
    colors = colormap(distances.detach().cpu().numpy())[:, :3]  # Get the RGB values
    return torch.from_numpy(colors)

def render_point_cloud(colors, camera, point_cloud):
    # Define the settings for rasterization and shading
    raster_settings = PointsRasterizationSettings(
        image_size=256, radius = 0.01, points_per_pixel = 10)
    rasterizer = PointsRasterizer(
        cameras=camera, raster_settings=raster_settings)
    compositor = AlphaCompositor()
    renderer = PointsRenderer(rasterizer=rasterizer,compositor=compositor)
    
    images = renderer(point_cloud)
    
    return images.clamp(0.0, 1.0)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0:
            print(f"{name:30s} {num_params / 1e6:.2f}M")

class Trainer():
    """Trainer Class containing train, validation, calibration"""
    def __init__(self, cfg, device):
        """Init config"""
        self.cfg = cfg
        self.device = device
        self.val_dataset = ShapeNetRenderedDataset(self.cfg, "val")
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, 
            num_workers=0, collate_fn=custom_collate_fn)

    def validate(self):
        """
        Args:
            model: model to train
            train_loader: training data loader
            optimizer: optimizer
            epoch: current epoch
            device: cuda or cpu
        """
        # torch.cuda.empty_cache()
        pbar = tqdm(self.val_dataloader)
        for i, (images, data) in enumerate(pbar):
            start_time = time.time()
            # Load data
            imgs = torch.stack([
                img.to(self.device) for img in images], dim=0)
            depth = torch.stack([
                d.to(self.device) for d in data['depth']], dim=0)
            pcd = torch.stack([
                p.to(self.device) for p in data['pcd']],dim=0)[0][0].unsqueeze(0)
            pcd_center = torch.stack([
                p.to(self.device) for p in data['pcd_center']],dim=0)
            pcd_radius = torch.stack([
                p.to(self.device) for p in data['pcd_radius']],dim=0)
            apcd = torch.stack([
                p.to(self.device) for p in data['apcd']],dim=0)[0][0].unsqueeze(0)
            pose = data['pose'][0]
            # print(pose.shape)
            R = torch.tensor(pose[:3,:3]).unsqueeze(0)
            T = torch.tensor(pose[:3,3]).unsqueeze(0)
            camera = FoVPerspectiveCameras(device=self.device, R=R, T=T).to(self.device)
            pcd_raster_settings = PointsRasterizationSettings(
                image_size=256, radius = 0.01, points_per_pixel = 10)
            pcd_rasterizer = PointsRasterizer(
                cameras=camera, raster_settings=pcd_raster_settings)
            pcd_renderer = PointsRenderer(
                rasterizer=pcd_rasterizer, compositor=AlphaCompositor())
            pcdd = apcd
            colors = color_points_white(pcdd).unsqueeze(1).expand(-1, pcdd.shape[1], -1)
            # print("pcd, colors",pcdd.shape, colors.shape)
            pcd_point_clouds = Pointclouds(points=pcdd, features=colors)
            pcd_images = pcd_renderer(pcd_point_clouds)
            pcd_images = pcd_images.clamp(0.0, 1.0)
            # print(pcd_images.shape)
            plt.imsave(f"./output/{i}_apcd.png", pcd_images.squeeze().detach().cpu().numpy())
            # plt.imsave(f"./output/{i}_pred.png", )

            img = imgs[0][0]
            # print(img.shape)
            mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
            img = img * std[:, None, None] + mean[:, None, None]
                        
            img = img.clamp(0.0, 1.0).detach().cpu().numpy().transpose(1,2,0)
            print(img.shape)
            plt.imsave(f"./output/{i}_img.png", img)


            R_ = torch.eye(3).unsqueeze(0).to(self.device)  # Identity matrix for rotation
            T_ = torch.zeros(1, 3).to(self.device)         # Zero translation vector

            camera = FoVPerspectiveCameras(
                device=self.device, R=R_, T=T_).to(self.device)
            pcd_raster_settings = PointsRasterizationSettings(
                image_size=256, radius = 0.01, points_per_pixel = 10)
            pcd_rasterizer = PointsRasterizer(
                cameras=camera, raster_settings=pcd_raster_settings)
            pcd_renderer = PointsRenderer(
                rasterizer=pcd_rasterizer, compositor=AlphaCompositor())
            pcdd = pcd
            colors = color_points_white(pcdd)\
                .unsqueeze(1).expand(-1, pcdd.shape[1], -1)
            # print("pcd, colors",pcdd.shape, colors.shape)
            pcd_point_clouds = Pointclouds(points=pcdd, features=colors)
            pcd_images = pcd_renderer(pcd_point_clouds)
            pcd_images = pcd_images.clamp(0.0, 1.0)
            plt.imsave(f"./output/{i}_vpcd.png", pcd_images.squeeze().detach().cpu().numpy())



def main(cfg, rank):
    output_dir = cfg['5_output_dir']
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # If it exists, delete all files inside
    else:
        shutil.rmtree(output_dir) # Removes all the subdirectories!
        os.makedirs(output_dir)

    trainer = Trainer(cfg, rank)    
    trainer.validate()


if __name__ == '__main__':
    """Main function"""
    # hvd.init()

    # torch.cuda.empty_cache()
    mp.set_start_method('spawn')

    # Load configuration from YAML file
    with open("./config/config.yml", 'r') as f:
        cfg = yaml.safe_load(f)

    # main(cfg, rank=hvd.rank())
    main(cfg, rank=torch.device('cpu'))
