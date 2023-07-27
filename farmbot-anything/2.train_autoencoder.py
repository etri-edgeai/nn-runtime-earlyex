import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from argparse import ArgumentParser
from tqdm import tqdm
import pytorch3d as p3d
from pytorch3d.loss import chamfer_distance
from modules.data.srendered import ShapeNetRenderedDataset, custom_collate_fn
from modules.autoencoder import PCDAutoEncoder
import torch.multiprocessing as mp
    
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
        self.device = torch.device(cfg['2_device'])
        self.dataset = ShapeNetRenderedDataset(self.cfg, "train")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=cfg['2_batch_size'], 
            shuffle=True, num_workers=cfg['2_num_workers'],
            collate_fn=custom_collate_fn)
        
        self.val_dataset = ShapeNetRenderedDataset(self.cfg, "val")
        self.val_dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=16, shuffle=False, num_workers=16,
                collate_fn=custom_collate_fn)
                
        self.model = PCDAutoEncoder(
            embedding_dims=cfg['2_embedding_dims'],
            pcd_samples=2048).to(self.device)
            
        count_parameters(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg['2_learning_rate'])
        self.epoch = cfg['2_epochs']
        self.mse_loss = nn.MSELoss()

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
        for epoch in range(self.epoch):
            pbar = tqdm(self.dataloader)
            total_loss = 0.0
            num_batches = 0
            for i, (img, target) in enumerate(pbar):
                self.optimizer.zero_grad()
                target = {k: v.to(self.device) for k, v in target.items()}
                output = self.model(target['pcd'])
                loss = chamfer_distance(output, target['pcd'])[0] + \
                        self.mse_loss(output, target['pcd'])
                total_loss += loss.item()
                num_batches += 1
                loss.backward()
                self.optimizer.step()
            avg_loss = total_loss / num_batches
            print(f"[{epoch}/{cfg['epochs']}]Loss: {avg_loss:.4f}")
            if epoch % 10 == 0:
                print("saving model...")
                torch.save(
                    self.model.decoder.state_dict(), cfg['2_pcd_checkpoints'])
            if epoch % 2 == 0:
                self.valid(epoch)

    def valid(self, test_index):
        """
        Args:
            test_index: Index of the test data in the dataset
        """
        self.model.eval()  # Set the model to evaluation mode
        pbar = tqdm(self.val_dataloader)
        total_loss = 0.0
        num_batches = 0
        for i, (img, target) in enumerate(pbar):
            with torch.no_grad():  # Do not compute gradient for this block
                target = {k: v.to(self.device) for k, v in target.items()}
                # Forward the test data through the model
                output = self.model(target['pcd'])

                # Compute the Chamfer distance 
                # between the original and reconstructed point clouds
                loss = chamfer_distance(
                    output, target['pcd'])[0] + \
                        self.mse_loss(output, target['pcd'])
                total_loss += loss.item()
                num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"[{test_index}/{cfg['epochs']}]Validation Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    """Main function"""
    torch.cuda.empty_cache()
    mp.set_start_method('spawn')
    # Load configuration from YAML file
    with open("./config/config.yml", 'r') as f:
        cfg = yaml.safe_load(f)
    trainer = Trainer(cfg)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("terminating training")
    finally:
        print("saving model...")
        torch.save(
            trainer.model.decoder.state_dict(), cfg['2_pcd_checkpoints'])

    original_pcd, reconstructed_pcd = trainer.validate(0)
