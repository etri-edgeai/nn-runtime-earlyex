import torch
from model.backbone.resnet import resnet18, resnet34, resnet50

def get_backbone(cfg):
    print(cfg['backbone'])
    model = None
    if cfg['backbone'] == "resnet18":
        model = resnet18(num_classes=cfg['num_class'])
    elif cfg['backbone'] == "resnet34":
        model = resnet34(num_classes=cfg['num_class'])
    elif cfg['backbone'] == "resnet50":
        model = resnet50(num_classes=cfg['num_class'])
    print("try to get pretrained backbone")
    try:
        model.load_state_dict(torch.load(cfg['backbone_path']),strict=False)

    except RuntimeError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)

    return model

