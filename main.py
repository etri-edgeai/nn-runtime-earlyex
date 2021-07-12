import argparse
import time

from utils.trainer import Trainer



def backbone_training(args, trainer, epochs = 30):
    print("Trainer backbone model...")
    try:
        for epoch in range(1, epochs):
            trainer.backbone_training(epoch)
            trainer.scheduler.step()
            trainer.backbone_validation(epoch)
    except KeyboardInterrupt:
        print("Skipping baseline training")

    return trainer

def backbone_validation(args, trainer):
    trainer.model.backbone.eval()
    trainer.backbone_validation(0)

def branch_training(args, trainer, epochs = 30):
    print("Trainer branch init...")
    trainer.branch_init(args)

    print("1. fine-tuning branch...")
    try:
        for epoch in range(1, epochs):
            trainer.branch_tuning(epoch)
            trainer.scheduler.step()
            trainer.branch_valid()
    except :
        print("Skipping baseline training")

    ent_list , acc_list = trainer.branch_valid()
    return ent_list, acc_list

def measure_time(args, trainer , endd):
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

def set_gates(args, trainer, gates=[], thresholds=[]):
    print("Setting gates & threshold")    
    for n in range(trainer.model.ex_num):
        trainer.model.set_branch(n,False)

    assert len(gates) == len(thresholds)

    for n, g in enumerate(gates):
        trainer.model.set_branch(g, True)
        trainer.model.exactly[g].threshold = thresholds[n]

def test_gates(args, trainer,endd= 10000):
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
    parser.add_argument('--pretrained', type=str, dest='pretrained',    default = "./my_trained/")
    parser.add_argument('--best'    ,   type=str, dest='best',          default = "./checkpoints/")
    parser.add_argument('--backbone',   type=str, dest='backbone',      default = 'resnet18')
    parser.add_argument('--dataset',    type=str, dest='dataset',       default = 'cifar10')
    parser.add_argument('--dali',       type=bool,dest='dali',          default = False)
    parser.add_argument('--data_dir',   type=str, dest='data_dir',      default = '/home/jyp/data/imagenet/')
    parser.add_argument('--epoch',      type=int, dest='epoch',         default = 30)
    parser.add_argument('--device',     type=str, dest='device',        default = 'cuda')
    parser.add_argument('--num_class',  type=int, dest='num_class',     default = 10)
    parser.add_argument('--batch_size', type=int, dest='batch_size',    default = 256)
    parser.add_argument('--workers',    type=int, dest='workers',       default = 4)
    parser.add_argument('--img_size',   type=int, dest='size',          default = 32)
    parser.add_argument('--lr',         type=float,                     default = 0.0001)
    
    args = parser.parse_args()
    args.best = args.best +args.backbone+"_best.pth"
    args.save = args.pretrained +args.backbone+"_"+args.dataset+"_best.pth"

    print("Trainer init...")
    trainer = Trainer(args)

    backbone_training(args, trainer, args.epoch)

    backbone_validation(args, trainer)

    branch_training(args, trainer, args.epoch)

    measure_time(args,trainer,10000)

    set_gates(args,trainer,[5],[0.15])

    test_gates(args,trainer,10000)





















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