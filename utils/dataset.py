import torchvision
import torchvision.transforms as transforms
import torch

def get_dataloader(cfg):
    
    trainset, testset = get_dataset(cfg)      
    train_loader   = torch.utils.data.DataLoader(trainset, batch_size= cfg['batch_size'], shuffle=True,  num_workers=cfg['workers'])
    val_loader     = torch.utils.data.DataLoader(testset,  batch_size= cfg['batch_size'], shuffle=False, num_workers=cfg['workers'])
    test_loader    = torch.utils.data.DataLoader(testset,  batch_size= 1,               shuffle=True,    num_workers=cfg['workers'])
    return train_loader, val_loader, test_loader, trainset, testset

def get_dataset(cfg):

    if cfg['path']['dataset'] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),])

        trainset = torchvision.datasets.CIFAR10(
            root = cfg['path']['dataset_path'], train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root = cfg['path']['dataset_path'], train=False, download=True, transform=transform_test)

    elif cfg['path']['dataset']  == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root = cfg['path']['dataset_path'], train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(
            root = cfg['path']['dataset_path'], train=False, download=True, transform=transform_test)

    elif cfg['path']['dataset']  == 'mnist':
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        trainset = torchvision.datasets.MNIST(cfg['path']['dataset_path'], train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(cfg['path']['dataset_path'], train=False, transform=transform)


    return trainset, testset
        
    