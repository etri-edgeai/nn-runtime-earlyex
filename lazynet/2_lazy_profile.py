
"""LazyNet Model Converter \
    from Pytorch to ONNX, TFLite, and Pytorch Mobile
"""
import torch
import torchvision
import model
from utils import *
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as  np
import yaml
from model.lazy import LazyNet
from argparse import ArgumentParser
from data import get_dataset
from tqdm import tqdm
import time
import utils
from torch.profiler import profile, record_function, ProfilerActivity

def parse_args():
    """Argument Parser function"""
    parser = ArgumentParser(
        description='convertion to onnx, tf, tflite')
    parser.add_argument('-c', type=str, default="./cfgs/base.yml",
                        help='model name (default: efficientnet-b0)')
    return parser.parse_args()

class Trainer():
    """Trainer Class containing train, validation, calibration"""
    def __init__(self, cfg):
        # Init config
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        
        # Get Dataset
        loaders = get_dataset(cfg)
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]

        # Set Model
        self.model = LazyNet(cfg).to(self.device)
        # filename = '{}_{}_{}.pth'.format(
        #     cfg['dataset'], cfg['backbone'], cfg['mode'])
        # self.model.load_state_dict(
        #     torch.load(cfg['save-dir']+filename), strict=False)
    
    def profile(self):
        self.model.eval()
        self.model.train_mode=False
        # input = torch.randn(1, 3, 128, 128).to(self.device)

        pbar = tqdm(self.test_loader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device)

            #     with profile(
            #         activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
            #         record_shapes=True) as prof:
            #         with record_function("model_inference"):
            #             self.model(images)
            # print(prof.key_averages().table(sort_by="cuda_time_total"))

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True, profile_memory=True) as prof:
                with record_function("model_inference"):
                    self.model(images)
            # print(
            #     prof.key_averages().table(
            #         sort_by="cuda_time_total", row_limit=10))

            # with profile(
            #     activities=[ProfilerActivity.CPU], 
            #     profile_memory=True, record_shapes=True) as prof:
            #     self.model(input)
            if i == 1000:
                break
        print(
            prof.key_averages().table(
                sort_by="cuda_memory_usage"))


def main():
    """convert pytorch model to:
    pytorch mobile
    pytorch mobile optimized for mobile
    onnx
    tensorflow
    tflite
    """
    args = parse_args()
    cfg = read_yaml(args.c)
    trainer = Trainer(cfg)
    trainer.profile()



if __name__ == '__main__':
    main()
