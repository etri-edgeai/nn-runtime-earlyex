import torchvision
import torchvision.transforms as transforms
import torch
import yaml
from pytorch_metric_learning import distances, losses, miners, reducers, testers, reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
import faiss 
import torch.nn as nn
import os
from .model.backbone.resnet import resnet18

def config(str):
    f = open(str, 'r')
    cfg = yaml.safe_load(f)

    for n in cfg['combine_dir']:
        print(n)
        cfg[n] = ''.join(cfg['combine_dir'][n])
        os.makedirs(cfg[n], exist_ok = True)

    for n in cfg['combine_path']:
        cfg[n] = ''.join(cfg['combine_path'][n])

    return cfg

def get_dataset(cfg):
    if cfg['dataset'] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.Resize((cfg['img_size'], cfg['img_size'])),
            transforms.ToTensor(),            
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2471, 0.2435, 0.2616))
            ])
      
        transform_test = transforms.Compose([
            transforms.Resize(size=(cfg['img_size'], cfg['img_size'])),            
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2471, 0.2435, 0.2616)
                )])

        trainset = torchvision.datasets.CIFAR10(
            root = cfg['dataset_dir'], 
            train=True, 
            download=True, 
            transform=transform_train
            )
        testset = torchvision.datasets.CIFAR10(
            root = cfg['dataset_dir'], 
            train=False, 
            download=True, 
            transform=transform_test
            )

    elif cfg['dataset']  == "cifar100":
        transform_train = transforms.Compose([
            transforms.Resize((cfg['img_size'], cfg['img_size'])),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        transform_test = transforms.Compose([
            transforms.Resize((cfg['img_size'], cfg['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

        trainset = torchvision.datasets.CIFAR100(
            root = cfg['dataset_dir'], 
            train=True, 
            download=True, 
            transform=transform_train)

        testset = torchvision.datasets.CIFAR100(
            root = cfg['dataset_dir'], 
            train=False, 
            download=True, 
            transform=transform_test)

    elif cfg['dataset']  == 'mnist':
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        trainset = torchvision.datasets.MNIST(
            cfg['dataset_dir'], 
            train=True, 
            download=True, transform=transform)
        
        testset = torchvision.datasets.MNIST(
            cfg['dataset_dir'], 
            train=False, 
            transform=transform)

    else:
        print("undefined, get your own dataset")

    return trainset, testset




def get_dataloader(cfg, select="train", train=None,val=None, test=None):

    trainset, testset = get_dataset(cfg) 


    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size= cfg['batch_size'], 
        shuffle=True,  
        num_workers=cfg['workers'],
        pin_memory=True)        
    
    val_loader = torch.utils.data.DataLoader(
        testset,  
        batch_size= cfg['batch_size'], 
        shuffle=False, 
        num_workers=cfg['workers'],
        pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        testset,  
        batch_size= 1, 
        shuffle=False, 
        num_workers=cfg['workers'],
        pin_memory=True)
    
    return train_loader, val_loader, test_loader



################################################################################



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count