import yaml
import argparse
from early_ex.utils import *
from early_ex.model.devour import DevourModel
from early_ex.model.backbone import get_backbone
from early_ex.trainer.dme_branch import DMEBranchTrainer
# from early_ex.trainer.ce_branch import DCEBranchTrainer
from tqdm import tqdm


print("Devour & Branch Trainer v0.9")

cfg = config("./configs/base.yml")
backbone = get_backbone(cfg)

model = DevourModel(cfg, N=cfg['num_exits'])
model.devour(backbone, cfg['backbone'])
trainset, testset = get_dataset(cfg)
self = DMEBranchTrainer(model, cfg)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# self.dcriterion = losses.SupConLoss(temperature=0.05)
self.dcriterion = losses.TripletMarginLoss(margin=0.05)
self.ccriterion = nn.CrossEntropyLoss()
# self.miner = miners.UniformHistogramMiner()
self.miner = miners.MultiSimilarityMiner()
self.optimizer   = torch.optim.Adam(model.parameters(), lr = 0.0001)
self.scheduler   = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10, 15, 20], gamma=0.5)

print("--Freeze Head and Body Layer...")
for n, m in self.model.feats.named_parameters():
    m.requires_grad_ = False
for n, m in self.model.head_layer.named_parameters():
    m.requires_grad_ = False
for n, m in self.model.fetc.named_parameters():
    m.requires_grad_ = False

for n, m in self.model.exactly.named_parameters():
    m.requires_grad_ = True
for n, m in self.model.tail_layer.named_parameters():
    m.requires_grad_ = False

print('criterion: Triplet Margin Loss')
print('optimizer: Adam, lr: ', self.cfg['lr'])
print('scheduler = multistep LR')
# Branch training


self.train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=cfg['batch_size'], 
        shuffle=True,  
        num_workers=cfg['workers'],
        pin_memory=True) 

self.val_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size=cfg['batch_size'], 
        shuffle=False,  
        num_workers=cfg['workers'],
        pin_memory=True) 

model = model.to(cfg['device'])



ex_num = self.model.n
num_class = self.cfg['num_class']

epochs = 50
print("Train")

for n in range(ex_num):
    self.model.exactly[n].requires_grad = True
    self.model.exactly[n].cross = False
    self.model.exactly[n].projectt = True
    self.model.exactly[n].gate = False
    
    
for epoch in range(epochs): 
    train_tbar = tqdm(self.train_loader)
    cnts = []
    embs = []
    self.model.nn = []
    for n in range(ex_num):
        self.model.nn.append(NN())
        cnts.append(torch.zeros(num_class).to(self.device))
        embs.append(torch.zeros(num_class, self.cfg['contra']['projection']).to(self.device))

    model.train()
    for i, data in enumerate(train_tbar):
        train_loss = torch.zeros([ex_num]).to(self.device)
        input = data[0].to(self.device)
        label = data[1].to(self.device)

        pred = self.model.forward(input)

        for n in range(ex_num):
            m = self.model.exactly[n]
            hard_pairs = self.miner(m.proj, label)
            train_loss[n] = self.dcriterion(m.proj, label, hard_pairs)

            for y in range(self.cfg['num_class']):
                split = m.proj[label == y].detach().cuda()
                cnts[n][y] += split.shape[0]
                embs[n][y] += torch.sum(split, dim=0).cuda()

        self.optimizer.zero_grad()
        loss = torch.sum(train_loss)    
        loss.backward()
        self.optimizer.step()

    print(embs[0][0])
    print(cnts[0][0])
    for n in range(ex_num):
        for y in range(num_class):
            embs[n][y] = torch.div(embs[n][y], cnts[n][y].item())
    print(embs[0][0])

    for n in range(ex_num):
        X = torch.nn.functional.normalize(embs[n])       
        Y = torch.linspace(0, self.cfg['num_class']-1, self.cfg['num_class'])
        print(Y)
        self.model.nn[n].train(X, Y)

    print("Validation")
    model.eval()
    total = 0
    corrects = []
    val_loss = []
    preds = []
    val_tbar = tqdm(self.val_loader)

    for n in range(model.n + 1):
        corrects.append(0.0)
        val_loss.append(0.0)
        preds.append(torch.zeros(0))

    with torch.no_grad():
        for i, (input, label) in enumerate(val_tbar):
            input = input.to(cfg['device'])
            label = label.to(cfg['device'])
            pred = model.forward(input)
            total += label.size(0)

            for n in range(model.n):
                m = model.exactly[n]
                proj = m.proj.detach().cuda()
                pred = self.model.nn[n].predict(proj).int().to(self.device) 
                corrects[n] += pred.eq(label).sum().item()

        for n in range(model.n):
            m = model.exactly[n]
            acc = 100.* corrects[n] / total
            val = val_loss[n] / total
            print("Branch: {}, acc: {:.2f}, val_loss: {:.4f}"
            .format(n, acc, val))
        acc = 100.* corrects[model.n] / total
        val = val_loss[model.n]        
    print("Output acc: {}, val_loss: {}".format(acc, val))
    torch.cuda.empty_cache()

