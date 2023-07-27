import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import json
from argparse import ArgumentParser
from tqdm import tqdm
import torch.multiprocessing as mp
from modules.mask_rcnn import maskrcnn_resnet50_fpn_v2
import os
import sys
import horovod.torch as hvd
from torch.utils.data.distributed import DistributedSampler
import time

cwd = os.getcwd()
sys.path.append(cwd)
from modules.data.coco import COCORenderedDataset, custom_collate_fn

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0:
            print(f"{name:30s} {num_params / 1e6:.2f}M")

class Trainer():
    """Trainer Class containing train, validation, calibration"""
    def __init__(self, cfg):
        """Init config"""
        self.cfg = cfg
        self.device = hvd.rank()
        self.dataset = COCORenderedDataset(self.cfg, "train")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=cfg['3_batch_size'], shuffle=False, 
            num_workers=cfg['3_num_workers'],  collate_fn=custom_collate_fn,
            sampler=DistributedSampler(
                self.dataset, num_replicas=hvd.size(), rank=hvd.rank()))

        jsonpath = cfg['1_train_json']
        with open(jsonpath, 'r') as f:
            self.coco_annotations = json.load(f)
        self.coco_categories = self.coco_annotations['categories']
        num_classes = len(self.coco_categories)+1 
        self.model = maskrcnn_resnet50_fpn_v2(num_classes=num_classes)
        try:
            print("Loading MaskRCNN checkpoint...")
            self.model.load_state_dict(torch.load(self.cfg['3_mrcnn_checkpoints']))
        except:
            print("No MaskRCNN checkpoint found, training from scratch...")
            
        count_parameters(self.model)
        self.epoch = self.cfg['3_epochs']

    def train(self):
        """
        Args:
            model: model to train
            train_loader: training data loader
            optimizer: optimizer
            epoch: current epoch
            device: cuda or cpu
        """
        self.model.train()
        self.model.to(self.device)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.cfg['3_learning_rate'])
        optimizer = hvd.DistributedOptimizer(
            optimizer, 
            named_parameters=self.model.named_parameters())
        hvd.broadcast_parameters(
            self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        for epoch in range(self.epoch):
            # pbar = tqdm(self.dataloader)
            total_loss = 0.0
            num_batches = 0
            # for i, (images, data) in enumerate(pbar):
            for i, (images, data) in enumerate(self.dataloader):
                start_time = time.time()
                imgs = [img.to(self.device) for img in images]
                masks = [mask.to(self.device) for mask in data['masks']]
                bboxes = [torch.stack(bbox_list).to(self.device)
                            for bbox_list in data['boxes']]
                labels = [torch.stack(label_list).to(self.device)
                            for label_list in data['labels']]
                
                targets = []
                for mask, bbox, label in zip(masks, bboxes, labels):
                    target = {}
                    target['masks'] = mask.unsqueeze(0)
                    target['boxes'] = bbox
                    target['labels'] = label
                    targets.append(target)
                loss = self.model(imgs, targets)[0]
                num_batches += 1
                losss = loss['loss_classifier'] + loss['loss_box_reg'] + \
                    loss['loss_objectness'] + loss['loss_rpn_box_reg']
                optimizer.zero_grad()
                losss.backward()
                optimizer.step()
                total_loss += losss.item()
                # self.optimizer.step()
                end_time = time.time()
                elapsed_time = end_time - start_time
                if hvd.rank() == 0 and i % 10 == 0:
                    total_iterations = len(self.dataloader)
                    remaining_iterations = total_iterations - i
                    remaining_time = remaining_iterations * elapsed_time
                    remaining_minutes = int(remaining_time // 60)
                    remaining_seconds = int(remaining_time % 60)
                    total_time = total_iterations * elapsed_time
                    total_minutes = int(total_time // 60)
                    total_seconds = int(total_time % 60)
                    print(f"[epoch: {epoch}], [{i}/{total_iterations}]")
                    print(f"Elapsed time: {elapsed_time:.2f} seconds")
                    print(f"Estimated total time: {total_minutes} minutes {total_seconds} seconds")
                    print(f"Estimated remaining time: {remaining_minutes} minutes {remaining_seconds} seconds")

            avg_loss = total_loss / num_batches
            print(f"[{i}/{self.cfg['3_epochs']}]Loss: {avg_loss:.4f}")
            torch.cuda.ipc_collect()
            if epoch % 5 == 0:
                print("Saving checkpoint...")
                torch.save(self.model.state_dict(), self.cfg['3_mrcnn_checkpoints'])



def main(cfg, rank):
    trainer = Trainer(cfg)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("terminating training")
    

if __name__ == '__main__':
    """Main function"""
    hvd.init()
    torch.cuda.empty_cache()
    torch.cuda.set_device(hvd.local_rank())
    mp.set_start_method('spawn')
    # Load configuration from YAML file
    with open("./config/config.yml", 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg, rank=hvd.rank())    
