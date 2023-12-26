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
from modules.model import FRUTNet
import torch.onnx
import onnx
import onnxruntime

cwd = os.getcwd()
sys.path.append(cwd)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from modules.dataloader import ShapeNetRenderedDataset, custom_collate_fn
# from modules.data.lrendered import ShapeNetRenderedDataset, custom_collate_fn
from torch.utils.data.distributed import DistributedSampler


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0:
            print(f"{name:30s} {num_params / 1e6:.2f}M")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    torch.cuda.empty_cache()

    mp.set_start_method('spawn')
    # Load configuration from YAML file
    with open("./config/config.yml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    

    model = FRUTNet(cfg, device = torch.device('cuda:0'))
    model.to(torch.device('cuda:0'))
    model.load_state_dict(torch.load( cfg['save_checkpoints']))
    model.eval()
    model.to_test()
    dataset = ShapeNetRenderedDataset(cfg, "train")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, 
        num_workers=cfg['num_workers'], collate_fn=custom_collate_fn)
    input = next(iter(dataloader))
    input = input[0][0].to(torch.device('cuda:0'))
    depth = input[1]['depth'].to(torch.device('cuda:0'))

    torch.onnx.export(model, (input, depth), "frutnet.onnx", verbose=False, \
                      opset_version=14, input_names=['input', 'depth'], \
                        output_names=['output'])
    onnx_model = onnx.load("frutnet.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("frutnet.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)
