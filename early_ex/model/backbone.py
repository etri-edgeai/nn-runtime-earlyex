from torchvision.models import efficientnet, \
    mobilenet_v2, vgg, inception_v3, resnet, \
        wide_resnet50_2, wide_resnet101_2

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

    elif m == 'vgg_19_bn':
        backbone = vgg.vgg19_bn(pretrained=True)

    elif m == 'wide_resnet50_2':
        backbone = wide_resnet50_2(pretrained=True)
    
    elif m == 'wide_resnet101_2':
        backbone = wide_resnet101_2(pretrained=True)

    return backbone

    # elif model_type == 'inception':
    #     backbone = inception_v3(pretrained=True)
        
    # elif model_type == 'vgg':
    #     backbone = vgg16_bn(pretrained=True)