import torch
from model.backbone.resnet import resnet18, resnet34

def get_backbone(cfg):
    print(cfg['backbone'])
    model = None
    if cfg['backbone'] == "resnet18":
        model = resnet18()
    elif cfg['backbone'] == "resnet34":
        model = resnet34()

    print("try to get pretrained backbone")
    try:
        if cfg['pretrained'] is not None:
            model.load_state_dict(torch.load('./checkpoints/state_dicts/'+cfg['backbone']+'.pt'),strict=False)

    except RuntimeError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)

    return model

