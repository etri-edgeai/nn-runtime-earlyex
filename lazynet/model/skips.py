from torch import nn
from model.cbrelu import CBRelu

class Skips(nn.Module):
    """Skips module"""
    def __init__(self, cfg, input, id):
        """Initialization function"""
        super().__init__()
        # Common configs
        self.cfg = cfg
        self.match_fc = cfg['__fc_features__']
        batch, channels, width, height = input.shape   
        
        self.skips = nn.Sequential(
            CBRelu(channels, self.match_fc//8, 1, 1, 0),
            CBRelu(self.match_fc//8, self.match_fc, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """Forward Function"""
        x = self.skips(input)
        return x
