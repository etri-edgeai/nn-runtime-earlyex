import argparse
import time
import yaml
from utils.trainer import Trainer



def backbone_training(cfg, trainer, epochs = 30):

    print("Trainer backbone model...")
    try:
        for epoch in range(1, epochs):
            trainer.backbone_training(epoch)
            trainer.scheduler.step()
            trainer.backbone_validation(epoch)
    except KeyboardInterrupt:
        print("Skipping baseline training")

    return trainer

def backbone_validation(cfg, trainer):
    trainer.model.backbone.eval()
    trainer.backbone_validation(0)

def branch_training(cfg, trainer):
    print("Trainer branch init...")
    trainer.branch_init(cfg)

    print("1. fine-tuning branch...")
    try:
        for epoch in range(1, cfg['branch_training']['epoch']):
            trainer.branch_tuning(epoch)
            trainer.scheduler.step()
            trainer.branch_valid()
    except :
        print("Skipping baseline training")

    ent_list , acc_list = trainer.branch_valid()
    return ent_list, acc_list

def measure_time(cfg, trainer , endd):
    endd = endd
    time_list = []
    print("Time checking each early branches")
    try:
        for n in range(0, trainer.model.ex_num-1):
            for m in range(trainer.model.ex_num):
                trainer.model.set_branch(m,False)
            trainer.model.backbone.eval()
            print(trainer.model.exnames[n])
        
            print("open gate ", n)
            trainer.model.set_exit(n+1, True)
            trainer.model.set_branch(n,True)
            start = time.time()
            trainer.test(end = endd)
            end = time.time()
            trainer.model.set_exit(n+1, False)
            spent = end - start
            print("time spent: {:.3f}".format(spent))
            trainer.model.exactly[n].time_cost = spent
            time_list.append(spent)
    except KeyboardInterrupt:
       print("Skipping single target testing")
    print("time_spent: ",time_list)
    
    print("Check entire time spent")
    try:
       trainer.model.backbone.eval()
       start = time.time()
       trainer.test(end = endd)
       end = time.time()
       spent = end - start
       time_list.append(spent)
    except KeyboardInterrupt:
       print("Skipping single target testing")

def set_gates(cfg, trainer, gates=[], thresholds=[]):
    print("Setting gates & threshold")    
    for n in range(trainer.model.ex_num):
        trainer.model.set_branch(n,False)

    assert len(gates) == len(thresholds)

    for n, g in enumerate(gates):
        trainer.model.set_branch(g, True)
        trainer.model.exactly[g].threshold = thresholds[n]

def test_gates(cfg, trainer,endd= 10000):
    print("Check testmode")
    try:
        trainer.model.test_mode = True
        trainer.model.backbone.eval()
        start = time.time()
        trainer.test(end = endd)
        end = time.time()
        spent = end - start
    except KeyboardInterrupt:
        print("Skipping single target testing")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = "./configs/init.yml")
    args = parser.parse_args()
    f = open(args.config, 'r')
    cfg = yaml.load(f)
    print(cfg['best'])
    print(cfg['best'])

    cfg['best_path'] = cfg['best'] + cfg['backbone'] + "_best.pth"
    cfg['save'] = cfg['pretrained'] +cfg['backbone']+"_"+cfg['dataset']+"_best.pth"

    print("Trainer init...")
    trainer = Trainer(cfg)

    backbone_training(cfg, trainer, cfg['backbone_training']['epoch'])

    backbone_validation(cfg, trainer)

    branch_training(cfg, trainer)

    measure_time(cfg,trainer,cfg['timed']['sample'])

    set_gates(cfg,
            trainer,
            cfg['set_gate']['gates'], 
            cfg['set_gate']['thresholds'])

    test_gates(cfg,trainer,cfg['timed']['sample'])





















#    print("3. prepare ")
#    trainer.model.t_tuning = True
#    for n in range(0, trainer.model.ex_num):
#        tim = time_list[n] / time_list[-1]
#        acc = acc_list[-1] / acc_list[n]
#        trainer.model.exactly[n].t_prepare(tim, acc , 0.1)

#    trainer.test()

#    print("4. fine-tuning threshold...")
#    for epoch in range(1, 5):
#        trainer.threshold_tuning(epoch, time_list, acc_list)
#        trainer.scheduler.step()
#        trainer.branch_valid()
            

if __name__ == "__main__":
    main()