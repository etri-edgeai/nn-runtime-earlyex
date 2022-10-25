"""Dataset Manager Module"""
import os
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy
from torchvision import datasets
from torch.utils.data import DataLoader

def get_dataset(cfg):
    """get dataset
    input
    ----
    cfg: cfg file must be loaded
    default: cifar10 dataset
    """
    trainset, testset = None, None

    if cfg['dataset'] == "cifar10":
    
        normalize = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2471, 0.2435, 0.2616))

        transform_train = transforms.Compose([
            transforms.Resize((cfg['img_size'], cfg['img_size'])),
            transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),            
            normalize])
        
        transform_test = transforms.Compose([
            transforms.Resize(size=(cfg['img_size'], cfg['img_size'])),            
            transforms.ToTensor(),
            normalize])
        
        trainset = datasets.CIFAR10(
            root = cfg['data-dir'], train=True, download=True,
            transform=transform_train)
        
        testset = datasets.CIFAR10(
            root = cfg['data-dir'], train=False, download=True,
            transform=transform_test)

    elif cfg['dataset']  == "cifar100":
        normalize = transforms.Normalize(
                (0.507, 0.4865, 0.4409), 
                (0.2673, 0.2564, 0.2761))
            
        transform_train = transforms.Compose([
            transforms.Resize(cfg['img_size']),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize])

        transform_test = transforms.Compose([
            transforms.Resize((cfg['img_size'], cfg['img_size'])),
            transforms.ToTensor(),
            normalize])

        trainset = torchvision.datasets.CIFAR100(
            root = cfg['data-dir'], 
            train=True, 
            download=True, 
            transform=transform_train)

        testset = torchvision.datasets.CIFAR100(
            root = cfg['data-dir'], 
            train=False, 
            download=True, 
            transform=transform_test)

    elif cfg['dataset']  == "imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
           
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(cfg['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transform_test = transforms.Compose([
            transforms.Resize((cfg['img_size'], cfg['img_size'])),
            transforms.ToTensor(),
            normalize])

        trainset = datasets.ImageFolder(
            root=cfg['train-dir'], 
            transform=transform_train)

        testset = datasets.ImageFolder(
            root=cfg['val-dir'], 
            transform=transform_test)


    if trainset != None and testset != None:
        trl, val, tel = get_dataloader(cfg, trainset, testset)
        return trl, val, tel
    
    return None

def get_dataloader(cfg, trainset, testset):
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
        shuffle=True, num_workers=cfg['workers'], pin_memory=True)

    valloader = DataLoader(testset, batch_size = cfg['batch_size'], 
        shuffle=False, num_workers = cfg['workers'], pin_memory = True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size = 1, shuffle=False,
        num_workers=32, pin_memory=True)

    return trainloader, valloader, testloader