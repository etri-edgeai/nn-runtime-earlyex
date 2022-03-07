from grpc import ssl_channel_credentials
import torch
from paramiko import SSHClient
from scp import SCPClient
import os
import torch
from tqdm import tqdm
from datetime import datetime
from early_ex.utils import config, get_dataloader, get_dataset
from early_ex.model.backbone import get_backbone
from early_ex.trainer.backbone import BackboneTrainer
from early_ex.model import Model
import argparse 
import sys

def main():
    print("Test backbone ")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = "./early_ex/configs/base.yml")
    args = parser.parse_args()
    cfg = config(args.config)


if __name__ == "__main__":
    main()