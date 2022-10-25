import copy
import torch
from torch import nn
from torchvision.models import efficientnet, \
    mobilenet_v2, vgg, inception_v3, resnet, \
        wide_resnet50_2, wide_resnet101_2, regnet

def get_backbone(cfg):
    """Get backbone based on string
    available backbones:
    resnet
    mobilenet_v2
    efficientnet_b0
    inception_v3
    vgg
    wide_resnet
    regnet
    """
    name = cfg['backbone']
    print("Backbone model type: ", name)
    backbone = None
    if   name == 'resnet18':
        backbone = resnet.resnet18(pretrained=True, progress=True)
        backbone.fc = torch.nn.Linear(
            backbone.fc.in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.fc.in_features

    elif name == 'resnet50':
        backbone = resnet.resnet50(pretrained=True, progress=True)
        backbone.fc = torch.nn.Linear(
            backbone.fc.in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.fc.in_features

    elif name == 'resnet101':
        backbone = resnet.resnet101(pretrained=True, progress=True)
        backbone.fc = torch.nn.Linear(
            backbone.fc.in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.fc.in_features

    elif name == 'mobilenet_v2':
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.classifier[-1].in_features

    elif name == 'efficientnet_b0':
        backbone = efficientnet.efficientnet_b0(pretrained=True)
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.classifier[-1].in_features

    elif name == 'efficientnet_b1':
        backbone = efficientnet.efficientnet_b1(pretrained=True)
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.classifier[-1].in_features

    elif name == 'efficientnet_b2':
        backbone = efficientnet.efficientnet_b2(pretrained=True)
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.classifier[-1].in_features

    elif name == 'efficientnet_b3':
        backbone = efficientnet.efficientnet_b3(pretrained=True)
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])

    elif name == 'efficientnet_b4':
        backbone = efficientnet.efficientnet_b4(pretrained=True)
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.classifier[-1].in_features

    elif name == 'efficientnet_b5':
        backbone = efficientnet.efficientnet_b5(pretrained=True)
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.classifier[-1].in_features

    elif name == 'efficientnet_b6':
        backbone = efficientnet.efficientnet_b6(pretrained=True)
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.classifier[-1].in_features

    elif name == 'efficientnet_b7':
        backbone = efficientnet.efficientnet_b7(pretrained=True)
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])
        cfg['__fc_features__'] = backbone.classifier[-1].in_features

    elif name == 'inception_v3':
        backbone = inception_v3(pretrained=True)

    elif name == 'vgg11_bn':
        backbone = vgg.vgg11_bn(pretrained=True)
        cfg['__fc_features__'] = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])

    elif name == 'vgg16_bn':
        backbone = vgg.vgg16_bn(pretrained=True)
        cfg['__fc_features__'] = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])

    elif name == 'vgg19_bn':
        backbone = vgg.vgg19_bn(pretrained=True)
        cfg['__fc_features__'] = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(
            backbone.classifier[-1].in_features, cfg['num_class'])
            
    elif name == 'wide_resnet50_2':
        backbone = wide_resnet50_2(pretrained=True)
    
    elif name == 'wide_resnet101_2':
        backbone = wide_resnet101_2(pretrained=True)

    elif name == 'regnet_y_400mf':
        backbone = regnet.regnet_y_400mf(pretrained=True)

    elif name == 'regnet_y_800mf':
        backbone = regnet.regnet_y_800mf(pretrained=True)
    
    elif name == 'regnet_y_1_6gf':
        backbone = regnet.regnet_y_1_6gf(pretrained=True)

    elif name == 'regnet_y_3_2gf':
        backbone = regnet.regnet_y_3_2gf(pretrained=True)

    elif name == 'regnet_y_8gf':
        backbone = regnet.regnet_y_8gf(pretrained=True)

    elif name == 'regnet_y_16gf':
        backbone = regnet.regnet_y_16gf(pretrained=True)

    elif name == 'regnet_y_32gf':
        backbone = regnet.regnet_y_32gf(pretrained=True)

    elif name == 'regnet_x_400mf':
        backbone = regnet.regnet_x_400mf(pretrained=True)

    elif name == 'regnet_x_800mf':
        backbone = regnet.regnet_x_800mf(pretrained=True)

    elif name == 'regnet_x_1_6gf':
        backbone = regnet.regnet_x_1_6gf(pretrained=True)

    elif name == 'regnet_x_3_2gf':
        backbone = regnet.regnet_x_3_2gf(pretrained=True)

    elif name == 'regnet_x_8gf':
        backbone = regnet.regnet_x_8gf(pretrained=True)
    return backbone

def get_plan(backbone, name):
    """direct body into parts"""
    print("----------------------------------------")
    print("Scouting Module...")
    body = {}

    if 'efficientnet' in name:
        e = len(backbone.features)
        body['head']            = backbone.features
        body['head_start']      = 0
        body['head_end']        = 0
        body['mid']             = backbone.features
        body['mid_start']       = 1
        body['mid_end']         = e
        body['tail']            = backbone
        body['tail_start']      = 1
        body['tail_end']        = 2

    elif 'mobilenet' in name:
        e = len(backbone.features)
        body['head']            = backbone.features
        body['head_start']      = 0
        body['head_end']        = 0

        body['mid']             = backbone.features
        body['mid_start']       = 1
        body['mid_end']         = e

        body['tail']            = backbone.classifier
        body['tail_start']      = 1
        body['tail_end']        = 2

    elif 'inception' in name:
        body['head']            = backbone
        body['head_start']      = 0
        body['head_end']        = 6

        body['mid']             = backbone
        body['mid_start']       = 7
        body['mid_end']         = 14

        body['tail']            = backbone
        body['tail_start']      = 16
        body['tail_end']        = 21
    
    elif 'vgg' in name:
        e = len(backbone.features)
        body['head']            = backbone.features
        body['head_start']      = 0
        body['head_end']        = 6

        body['mid']             = backbone.features
        body['mid_start']       = 7
        body['mid_end']         = e

        body['tail']            = backbone.classifier
        body['tail_start']      = 0
        body['tail_end']        = 6

    elif 'regnet' in name:
        e = len(backbone)
        body['head']            = backbone
        body['head_start']      = 0
        body['head_end']        = 0

        body['mid']             = backbone.trunk_output
        body['mid_start']       = 0
        body['mid_end']         = 3

        body['tail']            = backbone
        body['tail_start']      = 2
        body['tail_end']        = 3
    
    else:
        body['head']            = backbone
        body['head_start']      = 0
        body['head_end']        = 3

        body['mid']             = backbone
        body['mid_start']       = 4
        body['mid_end']         = 7

        body['tail']            = backbone
        body['tail_start']      = 8
        body['tail_end']        = 9
    return body

def get_modulelist(module, start=0, end=0):
    result = []
    counter = 0
    assert end >= start
    print("start: {}, end: {}".format(start, end))
    # print("----------------------------")
    for n, m in module.named_children():
        name = type(m).__name__
        if counter >= start and counter <= end:
            if name == "Linear":
                result.append(nn.Flatten())
            result.append(copy.deepcopy(m))
            print("idx: *\t",counter, "\t",name)
        else:
            print("idx: \t",counter,"\t",name)
        counter += 1
    return result