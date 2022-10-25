from torchvision.models import efficientnet, \
    mobilenet_v2, vgg, inception_v3, resnet, \
        wide_resnet50_2, wide_resnet101_2, regnet

def get_backbone(cfg):
    m = cfg['backbone']
    backbone = resnet.resnet18(num_classes= cfg['num_class'])
    print("Backbone model type: ", m)

    if m == 'resnet18':
        backbone = resnet.resnet18(num_classes= cfg['num_class'])

    elif m == 'resnet50':
        backbone = resnet.resnet50(num_classes= cfg['num_class'])

    elif m == 'resnet101':
        backbone = resnet.resnet101(num_classes= cfg['num_class'])

    elif m == 'mobilenet_v2':
        backbone = mobilenet_v2(num_classes= cfg['num_class'])

    elif m == 'efficientnet_b0':
        backbone = efficientnet.efficientnet_b0(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'efficientnet_b1':
        backbone = efficientnet.efficientnet_b1(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'efficientnet_b2':
        backbone = efficientnet.efficientnet_b2(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'efficientnet_b3':
        backbone = efficientnet.efficientnet_b3(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'efficientnet_b4':
        backbone = efficientnet.efficientnet_b4(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'efficientnet_b5':
        backbone = efficientnet.efficientnet_b5(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'efficientnet_b6':
        backbone = efficientnet.efficientnet_b6(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'efficientnet_b7':
        backbone = efficientnet.efficientnet_b7(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'inception_v3':
        backbone = inception_v3(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'vgg11_bn':
        backbone = vgg.vgg11_bn(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'vgg16_bn':
        backbone = vgg.vgg16_bn(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'vgg19_bn':
        backbone = vgg.vgg19_bn(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'wide_resnet50_2':
        backbone = wide_resnet50_2(num_classes= cfg['num_class'],pretrained=True)
    
    elif m == 'wide_resnet101_2':
        backbone = wide_resnet101_2(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_y_400mf':
        backbone = regnet.regnet_y_400mf(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_y_800mf':
        backbone = regnet.regnet_y_800mf(num_classes= cfg['num_class'],pretrained=True)
    
    elif m == 'regnet_y_1_6gf':
        backbone = regnet.regnet_y_1_6gf(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_y_3_2gf':
        backbone = regnet.regnet_y_3_2gf(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_y_8gf':
        backbone = regnet.regnet_y_8gf(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_y_16gf':
        backbone = regnet.regnet_y_16gf(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_y_32gf':
        backbone = regnet.regnet_y_32gf(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_x_400mf':
        backbone = regnet.regnet_x_400mf(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_x_800mf':
        backbone = regnet.regnet_x_800mf(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_x_1_6gf':
        backbone = regnet.regnet_x_1_6gf(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_x_3_2gf':
        backbone = regnet.regnet_x_3_2gf(num_classes= cfg['num_class'],pretrained=True)

    elif m == 'regnet_x_8gf':
        backbone = regnet.regnet_x_8gf(num_classes= cfg['num_class'],pretrained=True)

    return backbone
