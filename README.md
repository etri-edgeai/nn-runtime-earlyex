
# nn-runtime-earlyex
* A python based runtime tool for neural networks to achieve fast results using early exits

### 1. Distance based Early Exit Framework(DEX)

DEX is a framework that tries to insert distance based early exit classifiers.

#### Train backbone model

1. Set configure file ./earlyex/configs/base.yml

2. Run the following code to train backbone model:
> python train_backbone.py

#### Train Cross Entropy
1. Set configure file ./earlyex/configs/base.yml

2. Run the following code:
> python train_ce_branch.py

#### Train Metric based Classifier
1. Set configure file: ./earlyex/configs/base.yml

2. Run the following code:
> python train_me_branch.py


### 2. LazyNet: Lazy entry into Neural Networks(ICTC2022)

Usage:
1. Run the following code:

> python 1_train.py

base.yml is based on yaml, and contains config files


