import torch
import torch.nn as nn
import torch.nn.functional as F

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class MIMOUNet(nn.Module):
    def __init__(self, num_in=3, num_out=3, scale=1, multi_scale=False, filter=64):
        super(MIMOUNet, self).__init__()
        
        self.multi_scale = multi_scale
        self.encoder1 = nn.Sequential(
            nn.Conv2d(num_in, filter, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter, filter, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(filter, filter*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter*2, filter*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(filter*2, filter*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter*4, filter*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.eblock1 = EBlock(filter, num_res=4)
        self.eblock2 = EBlock(filter*2, num_res=4)
        self.eblock3 = EBlock(filter*4, num_res=4)
        
        self.dblock1 = DBlock(filter, num_res=4)
        self.dblock2 = DBlock(filter*2, num_res=4)
        self.dblock3 = DBlock(filter*4, num_res=4)
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(filter*4, filter*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter*2, filter*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(filter*2, filter, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter, filter, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(filter, num_out, kernel_size=3, padding=1)
        )
        
        if multi_scale:
            self.decoder3_out = nn.Conv2d(filter*2, num_out, kernel_size=3, padding=1)
            self.decoder2_out = nn.Conv2d(filter, num_out, kernel_size=3, padding=1)
            
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        e1 = self.eblock1(e1)
        e2 = self.eblock2(e2)
        e3 = self.eblock3(e3)
        
        d3 = self.dblock3(e3)
        d3 = self.decoder3(d3)
        
        d2 = d3 + e2
        d2 = self.dblock2(d2)
        d2 = self.decoder2(d2)
        
        d1 = d2 + e1
        d1 = self.dblock1(d1)
        d1 = self.decoder1(d1)
        
        if self.multi_scale:
            d3 = self.decoder3_out(d3)
            d2 = self.decoder2_out(d2)
            return [d1, d2, d3]
        else:
            return d1 