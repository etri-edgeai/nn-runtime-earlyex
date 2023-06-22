
# nn-runtime-earlyex
* A python based runtime tool for neural networks to achieve fast results using early exits

### 1. Distance based Early Exit Framework(DEX)

DEX is a framework that tries to insert distance based early exit classifiers.

#### Train backbone model
1. Run the following code to train backbone model:
```
python train_backbone.py
```

#### Train Cross Entropy
1. Run the following code:
```
python train_ce_branch.py
```

#### Train Metric based Classifier
```
python train_me_branch.py
```

### 2. LazyNet: Lazy entry into Neural Networks(ICTC2022)
Usage:
```
python 1_train.py
```

base.yml is based on yaml, and contains config files

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2021-0-00907, Development of Adaptive and Lightweight Edge-Collaborative Analysis Technology for Enabling Proactively Immediate Response and Rapid Learning).

