import os
import torch
import torchvision
from datetime import datetime
from early_ex.utils import *

def main():
    print("Standalone Test App for Early Exit")
    cfg = config("./test.yml")
    trainset, testset = get_dataset(cfg)

    test_loader = torch.utils.data.DataLoader(
        testset,  
        batch_size= 1, 
        shuffle=False, 
        num_workers=1,
        pin_memory=False)

    try:
        print("loading model for testing....")
        model = torch.jit.load('./checkpoints/model_scripted.pt')
    except RuntimeError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)
        print(" file not found! Maybe try training it first?")

    model.eval()
    acc, total = 0, 0 
    device = cfg['test_device']
    model.to(device)
    start = datetime.now()
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader):
            input = input.to(device)
            label = label.to(device)
            total += input.shape[0]
            pred = model.forward(input)
            _ , pred = torch.max(pred, 1)
            acc += pred.eq(label).sum().item()
        print("accuracy: ", acc/ total)
    date = datetime.now() - start
    print("time elapsed: {}".format(date))
if __name__ == "__main__":

    main()