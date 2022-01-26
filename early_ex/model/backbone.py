from torchvision.models import efficientnet, \
    mobilenet_v2, vgg, inception_v3, resnet, \
        wide_resnet50_2, wide_resnet101_2, regnet

def get_backbone(cfg):
    m = cfg['backbone']
    backbone = resnet.resnet18(cfg)
    print("Backbone model type: ", m)

    if   m == 'resnet18':
        backbone = resnet.resnet18(cfg)

    elif m == 'resnet50':
        backbone = resnet.resnet50(cfg)

    elif m == 'resnet101':
        backbone = resnet.resnet101(cfg)

    elif m == 'mobilenet_v2':
        backbone = mobilenet_v2(cfg)

    elif m == 'efficientnet_b0':
        backbone = efficientnet.efficientnet_b0(pretrained=True)

    elif m == 'efficientnet_b1':
        backbone = efficientnet.efficientnet_b1(pretrained=True)

    elif m == 'efficientnet_b2':
        backbone = efficientnet.efficientnet_b2(pretrained=True)

    elif m == 'efficientnet_b3':
        backbone = efficientnet.efficientnet_b3(pretrained=True)

    elif m == 'efficientnet_b4':
        backbone = efficientnet.efficientnet_b4(pretrained=True)

    elif m == 'efficientnet_b5':
        backbone = efficientnet.efficientnet_b5(pretrained=True)

    elif m == 'efficientnet_b6':
        backbone = efficientnet.efficientnet_b6(pretrained=True)

    elif m == 'efficientnet_b7':
        backbone = efficientnet.efficientnet_b7(pretrained=True)

    elif m == 'inception_v3':
        backbone = inception_v3(pretrained=True)

    elif m == 'vgg11_bn':
        backbone = vgg.vgg11_bn(pretrained=True)

    elif m == 'vgg16_bn':
        backbone = vgg.vgg16_bn(pretrained=True)

    elif m == 'vgg19_bn':
        backbone = vgg.vgg19_bn(pretrained=True)

    elif m == 'wide_resnet50_2':
        backbone = wide_resnet50_2(pretrained=True)
    
    elif m == 'wide_resnet101_2':
        backbone = wide_resnet101_2(pretrained=True)

    elif m == 'regnet_y_400mf':
        backbone = regnet.regnet_y_400mf(pretrained=True)

    elif m == 'regnet_y_800mf':
        backbone = regnet.regnet_y_800mf(pretrained=True)
    
    elif m == 'regnet_y_1_6gf':
        backbone = regnet.regnet_y_1_6gf(pretrained=True)

    elif m == 'regnet_y_3_2gf':
        backbone = regnet.regnet_y_3_2gf(pretrained=True)

    elif m == 'regnet_y_8gf':
        backbone = regnet.regnet_y_8gf(pretrained=True)

    elif m == 'regnet_y_16gf':
        backbone = regnet.regnet_y_16gf(pretrained=True)

    elif m == 'regnet_y_32gf':
        backbone = regnet.regnet_y_32gf(pretrained=True)

    elif m == 'regnet_x_400mf':
        backbone = regnet.regnet_x_400mf(pretrained=True)

    elif m == 'regnet_x_800mf':
        backbone = regnet.regnet_x_800mf(pretrained=True)

    elif m == 'regnet_x_1_6gf':
        backbone = regnet.regnet_x_1_6gf(pretrained=True)

    elif m == 'regnet_x_3_2gf':
        backbone = regnet.regnet_x_3_2gf(pretrained=True)

    elif m == 'regnet_x_8gf':
        backbone = regnet.regnet_x_8gf(pretrained=True)

    return backbone
