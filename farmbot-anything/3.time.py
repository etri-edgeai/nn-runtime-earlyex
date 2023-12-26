import json
import os
import sys
from argparse import ArgumentParser
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from pytorch3d.loss import chamfer_distance
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import onnx
# import horovod.torch as hvd
from modules.utils import *
cwd = os.getcwd()
sys.path.append(cwd)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from modules.dataloader import ShapeNetRenderedDataset, custom_collate_fn
# from modules.data.lrendered import ShapeNetRenderedDataset, custom_collate_fn
from modules.model import FRUTNet

# from modules.rgb_model import FRUTNet
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.structures import Pointclouds
import torch.onnx
import argparse
# from onnx_tf.backend import prepare
# import onnx2tf
# import onnxruntime
# import tensorflow as tf

# writer = SummaryWriter("./runs/FRUTNet/pcd")

from torch.profiler import profile, record_function, ProfilerActivity

class Trainer():
    """Trainer Class containing train, validation, calibration"""
    def __init__(self, cfg, device):
        """Init config"""
        self.cfg = cfg
        self.device = device
        self.cate_loss_weight   = 1.0
        self.ins_loss_weight    = 1.0
        self.pcd_loss_weight    = 1.0
        self.reg_loss_weight    = 1.0

        self.dataset = ShapeNetRenderedDataset(self.cfg, "train")
        print("Trainer.dataset: ", self.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=cfg['batch_size'], shuffle=True, 
            num_workers=cfg['num_workers'], collate_fn=custom_collate_fn)
        
        print("Trainer.dataloader: ", self.dataloader)
        self.val_dataset = ShapeNetRenderedDataset(self.cfg, "val")
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, shuffle=False,
            collate_fn=custom_collate_fn)

        print("loading model...")
        self.model = FRUTNet(self.cfg,  device = self.device)
        try:
            self.model.load_state_dict(torch.load(cfg['pretrain_checkpoints']))
        except:
            print("No FRUTNet checkpoint found, training from scratch...")
            
        self.model.to(self.device)
        self.epoch = self.cfg['epochs']
        print("Trainer.epoch: ", self.cfg['epochs'])
        print("Trainer.batch_size: ", self.cfg['batch_size'])
        print("Trainer.num_workers: ", self.cfg['num_workers'])
        print("Trainer.learning_rate: ", self.cfg['learning_rate'])
        print("Trainer.save_checkpoints: ", self.cfg['pretrain_checkpoints'])

    def validate(self, epoch):
        model = self.model.to(self.device)
        model.to_validate()
        pbar = tqdm(self.val_dataloader)
        running_loss        = 0.0
        running_inst_loss   = 0.0
        running_cate_loss   = 0.0
        running_reg_loss    = 0.0  
        running_pcd_loss    = 0.0

        for i, (images, data) in enumerate(pbar):
            imgs = torch.stack([img.to(self.device) for img in images], dim=0)
            depth = torch.stack([d.to(self.device) for d in data['depth']], dim=0)
            true_pcd = [p.to(self.device) for p in data['pcd']]
            masks  = [torch.stack(m).to(self.device) for m in data['masks']]
            bboxes = [torch.stack(b).to(self.device) for b in data['boxes']]
            labels = [torch.stack(l).to(self.device) for l in data['labels']]
            regress= [r.to(self.device) for r in data['regress']]
            imgs = imgs.squeeze(0)
            pose = data['pose'][0]

            los = self.model(imgs, depth, true_pcd, masks, bboxes, labels, regress)           

            # R = torch.tensor(pose[:3, :3]).unsqueeze(0)
            # T = torch.tensor(pose[:3,  3]).unsqueeze(0)
            # camera = FoVPerspectiveCameras(device=self.device, R=R, T=T).to(self.device)
            # pcd_raster_settings = PointsRasterizationSettings(
            #     image_size= 256, radius=0.01, points_per_pixel=10)
            # pcd_rasterizer = PointsRasterizer(
            #     cameras=camera, raster_settings=pcd_raster_settings)
            # pcd_renderer = PointsRenderer(
            #     rasterizer=pcd_rasterizer, compositor=AlphaCompositor())

            # true_pcd = true_pcd[0] 
            # # true_pcd_normalized = true_pcd - true_pcd.mean(dim=1, keepdim=True)
            # # true_pcd_normalized = true_pcd_normalized / torch.max(torch.norm(true_pcd_normalized, dim=1))

            
            # colors = color_points_white(true_pcd).unsqueeze(0).expand(-1, true_pcd.shape[-2], 3)

            # pcd_point_clouds = Pointclouds(points=true_pcd, features=colors)
            # pcd_images = pcd_renderer(pcd_point_clouds)
            # pcd_images = pcd_images.clamp(0.0, 1.0)
            # images =  pcd_images.permute(0,3,1,2).squeeze(0).detach().cpu().numpy()
            # writer.add_image('Image/true_pcd', images, i)
            # plt.imsave(f"./output/{i}_true_pcd.png", pcd_images.squeeze(0).detach().cpu().numpy())
            
            # pred_pcd = los['pcd_preds'][0]
            # colors = color_points_white(pred_pcd).unsqueeze(0).expand(-1, pred_pcd.shape[-2], 3)
            # pcd_point_clouds = Pointclouds(points=pred_pcd, features=colors)
            # pcd_images = pcd_renderer(pcd_point_clouds)
            # pcd_images = pcd_images.clamp(0.0, 1.0)
            # images =  pcd_images.permute(0,3,1,2).squeeze(0).detach().cpu().numpy()
            # writer.add_image('Image/pred_pcd', images, i)
            # plt.imsave(f"./output/{i}_pred_pcd.png", pcd_images.squeeze(0).detach().cpu().numpy())
            
            
            running_inst_loss   += los['inst_loss'].item()
            running_cate_loss   += los['cate_loss'].item()
            running_reg_loss    += los['reg_loss'].item()
            running_pcd_loss    += los['pcd_loss'].item()
            running_loss        += los['inst_loss'].item() \
                + los['cate_loss'].item() \
                    + los['reg_loss'].item() \
                        + los['pcd_loss'].item()

        # writer.add_scalar('pcd_val_inst_loss' , running_inst_loss / len(self.val_dataloader), epoch)
        # writer.add_scalar('pcd_val_cate_loss' , running_cate_loss / len(self.val_dataloader), epoch)
        # writer.add_scalar('pcd_val_reg_loss'  , running_reg_loss  / len(self.val_dataloader), epoch)
        # writer.add_scalar('pcd_val_pcd_loss'  , running_pcd_loss  / len(self.val_dataloader), epoch)
        # writer.add_scalar('pcd_val_loss'      , running_loss      / len(self.val_dataloader), epoch)
        print("pcd_val_inst_loss: ", running_inst_loss/len(self.val_dataloader))
        print("pcd_val_cate_loss: ", running_cate_loss/len(self.val_dataloader))
        print("pcd_val_reg_loss: ", running_reg_loss/len(self.val_dataloader))
        print("pcd_val_pcd_loss: ", running_pcd_loss/len(self.val_dataloader))
        print("pcd_val_loss: ", running_loss/len(self.val_dataloader))

        model.to_train()


if __name__ == '__main__':
    """Main function"""
    print("Fruit Ripeness & Utility Measure Tool using 3D Point Clouds Network(FRUTNet)")
    torch.cuda.empty_cache()
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Train FRUTNet'
    )

    parser.add_argument('-c', '--config', default='./config/config.yml', type=str,help='load config file')
    parser.add_argument('-f', '--fast_mode', action='store_true', default=False, help='fast mode')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset = ShapeNetRenderedDataset(cfg, "train")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, 
        num_workers=cfg['num_workers'], collate_fn=custom_collate_fn)
    device = torch.device("cpu")
    pbar = tqdm(dataloader)

    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    model = FRUTNet(cfg,  device = torch.device("cpu"))
    model.to_validate()
    if args.fast_mode:
        model.to_fast()

    for i, (images, data) in enumerate(pbar):
        imgs = torch.stack([img.to(device) for img in images], dim=0)
        depth = torch.stack([d.to(device) for d in data['depth']], dim=0)
        true_pcd = [p.to(device) for p in data['pcd']]
        masks  = [torch.stack(m).to(device) for m in data['masks']]
        bboxes = [torch.stack(b).to(device) for b in data['boxes']]
        labels = [torch.stack(l).to(device) for l in data['labels']]
        regress= [r.to(device) for r in data['regress']]
        imgs = imgs.squeeze(0)
        pose = data['pose'][0]
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    los = model(imgs, depth, true_pcd, masks, bboxes, labels, regress)           
                break
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")

    from ultralytics import YOLO

    model2 = YOLO("./checkpoints/yolov8m-seg.pt")
