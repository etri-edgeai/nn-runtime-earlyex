import torch
from model.backbone.resnet import resnet18, resnet34

def get_backbone(cfg):
    print(cfg['path']['backbone'])
    model = None
    if cfg['path']['backbone'] == "resnet18":
        model = resnet18()
    elif cfg['path']['backbone'] == "resnet34":
        model = resnet34()

    print("try to get pretrained backbone")
    try:
        model.load_state_dict(torch.load(cfg['backbone_path']),strict=False)

    except RuntimeError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)

    return model

