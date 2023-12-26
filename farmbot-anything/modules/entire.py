import torch
import torch.nn as nn
import torch.nn.functional as F

class CBRelu(nn.Module):
    """Block module class"""
    def __init__(
        self, 
        in_channels=256, 
        out_channels=256, 
        kernel_size=3,
        stride=1, padding=0, dilation=1, groups=1):
        """init function"""
        super(CBRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
        
    def forward(self, x):
        """forward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

class FeatureFusionEncoder(nn.Module):
    def __init__(self, cfg, in_channels=1280, out_channels=64):
        # E3RF RGBD and Detection Fusion 
        super(FeatureFusionEncoder, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']

        self.conv1 = CBRelu(
            in_channels=in_channels, out_channels=out_channels, \
                kernel_size=1, stride=1, padding=0, \
                dilation=1, groups=1)

        # Attention 설정
        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            CBRelu(out_channels, out_channels // 2, 1, 1, 0),
            CBRelu(out_channels // 2, out_channels, 1, 1, 0),
        )

        self.conv2 = CBRelu(
            in_channels=out_channels, out_channels=out_channels, \
                kernel_size=1, stride=1, padding=0, \
                dilation=1, groups=1)

        # Encoder 설정
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*64, 128),
            nn.ReLU(),
            nn.Linear(128, 2048),
        )

    def forward(self, x1, x2):
        # Feature Fusion 모듈 추론
        x1 = F.interpolate(x1, size=(32, 32), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(32, 32), mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2], dim=1)
        # Attention 적용
        x = self.conv1(x)
        att = self.atten(x)
        x = x + x * att
        x = self.conv2(x)
        out = self.encoder(x)
        out = out.view(-1, 2048)
        return out


class PCDDecoder(nn.Module):
    def __init__(self, embedding_dims= 512, pcd_samples = 2048):
        super(PCDDecoder, self).__init__()
        self.embedding_dims = embedding_dims
        self.pcd_samples    = pcd_samples
        self.fc1            = nn.Linear(embedding_dims, 512)
        self.fc2            = nn.Linear(512, 1024)
        self.fc3            = nn.Linear(1024, 3 * pcd_samples)


    def forward(self, x):
        """
        Args:
            x: (B, embedding_dims)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # (B, 3 * pcd_samples)
        x = x.view(-1, 3, self.pcd_samples) # (B, 3, pcd_samples)
        return x
    

class RGBDInput(nn.Module):
    def __init__(self, cfg):
        super(RGBDInput, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.img_size = cfg['img_size']
        self.batch_size = cfg['batch_size']
        self.num_class = 49
    
        self.rgbd = nn.Sequential(
            CBRelu(4, 64, 7, 2, 3),
            CBRelu(64, 128, 5, 2, 2),
            CBRelu(128, 256, 5, 2, 2),
            )

    def forward(self, input):
        x = self.rgbd(input)
        return x
    

class E3RFnet(nn.Module):
    """entire segmentation module"""
    def __init__(self, cfg, device):
        # E3RF 분석 모듈 초기화 설정
        super(E3RFnet, self).__init__()
        self.cfg            = cfg
        self.device         = cfg['device']
        self.img_size       = cfg['img_size']
        self.batch_size     = cfg['batch_size']       
        self.num_class      = 49
        # E3RF Backbone+FPN 모듈 선언
        self.backbone       = maskrcnn_resnet50_fpn_v2(num_classes=self.num_class)
        self.backbone.train()
        self.backbone.training = True        
        try:
            print("Loading MaskRCNN checkpoint...")
            self.backbone.load_state_dict(torch.load(self.cfg['seg_checkpoints']))
        except:
            print("No MaskRCNN checkpoint found, training from scratch...")
        
        # self.backbone.requires_grad_ = False
        # self.backbone.eval()
        # self.backbone.training = False
        
        self.rgbd_input = RGBDInput(cfg)
        self.encoder    = FeatureFusionEncoder(cfg)
        self.decoder    = PCDDecoder(embedding_dims= 2048, pcd_samples = 2048)

        try:
            print("Loading Decoder checkpoint...")
            self.decoder.load_state_dict(torch.load(self.cfg['pcd_checkpoints']))
        except:
            print("No Decoder checkpoint found, training from scratch...")

        self.training = True
        # E3RF Loss 모듈 선언

    
    def forward(self, rgb, depth, pcd=None, targets=None):
        """
        Args:
            rgb: (B, 3, H, W)
            depth: (B, 1, H, W)
        """
        B, C, H, W = rgb.shape

        # self.backbone.requires_grad_ = False
        # self.backbone.eval()
        # self.backbone.training = False
        # RGBD_Input 
        rgbd_input = torch.cat((rgb, depth.unsqueeze(1)), dim=1)
        rgbd_output = self.rgbd_input.forward(rgbd_input)
        # print("rgbd_output: ", rgbd_output.shape)
        det_loss, det, f = self.backbone.forward(rgb, targets=targets)
        f_0 = F.interpolate(
            f['0'], size=(32, 32), mode='bilinear', align_corners=False)
        f_1 = F.interpolate(
            f['1'], size=(32, 32), mode='bilinear', align_corners=False)
        f_2 = F.interpolate(
            f['2'], size=(32, 32), mode='bilinear', align_corners=False)
        f_3 = F.interpolate(
            f['3'], size=(32, 32), mode='bilinear', align_corners=False)
        det_output = torch.stack([f_0,f_1,f_2,f_3], dim=1)
        # print(det_output.shape)
        # det_output = torch.stack([d['masks'] for d in det], dim=1)
        # det_output = det_output.permute(1,0,3,4,2)
        det_output = det_output.view(B, -1, 32, 32)
        # print("det_output: ", det_output.shape)
        # print("detection: ", detection[0]['masks'].shape)
        # Encoder
        encoder_output = self.encoder.forward(rgbd_output, det_output)
        # print("encoder_output: ", encoder_output.shape)
        decoder_output = self.decoder.forward(encoder_output)
        # print("decoder_output: ", decoder_output.shape)
        decoder_output = decoder_output.permute(0,2,1)
        # print("decoder_output: ", decoder_output.shape)
        # print("pcd: ", pcd.shape)
        pcd = pcd.permute(0,2,1)

        if not self.training:
            return None, pcd, det
        else:
            loss_0 = chamfer_distance(decoder_output.float(), pcd.float())
            loss_1 = (det_loss['loss_mask'] + det_loss['loss_classifier'] + \
                det_loss['loss_box_reg'] + det_loss['loss_objectness'] + \
                det_loss['loss_rpn_box_reg'], None)
            loss = loss_0 + loss_1
            return loss, pcd, det