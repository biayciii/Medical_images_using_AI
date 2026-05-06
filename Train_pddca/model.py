import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class CrossAttention(nn.Module):
    
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_query, x_key_value):
        batch_size, C, width, height = x_query.size()
        
        proj_query = self.query_conv(x_query).view(batch_size, -1, width * height).permute(0, 2, 1)
        
        proj_key = self.key_conv(x_key_value).view(batch_size, -1, width * height)
        proj_value = self.value_conv(x_key_value).view(batch_size, -1, width * height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x_query
        return out

class DualStreamLateFusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_classes=1):
        super(DualStreamLateFusionUNet, self).__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1_1 = DoubleConv(in_channels, 64)
        self.enc1_2 = DoubleConv(64, 128)
        self.enc1_3 = DoubleConv(128, 256)
        self.enc1_4 = DoubleConv(256, 512)
        
        self.enc2_1 = DoubleConv(in_channels, 64)
        self.enc2_2 = DoubleConv(64, 128)
        self.enc2_3 = DoubleConv(128, 256)
        self.enc2_4 = DoubleConv(256, 512)

        self.bottleneck_ct = DoubleConv(512, 1024)
        self.bottleneck_mri = DoubleConv(512, 1024)

        self.cross_attention = CrossAttention(in_channels=1024)

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(2048, 1024)
        
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256, 64)

        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, ct, smri):
        
        c1 = self.enc1_1(ct)
        c2 = self.enc1_2(self.pool(c1))
        c3 = self.enc1_3(self.pool(c2))
        c4 = self.enc1_4(self.pool(c3))
        b_ct = self.bottleneck_ct(self.pool(c4))

        m1 = self.enc2_1(smri)
        m2 = self.enc2_2(self.pool(m1))
        m3 = self.enc2_3(self.pool(m2))
        m4 = self.enc2_4(self.pool(m3))
        b_mri = self.bottleneck_mri(self.pool(m4))

        attn_out = self.cross_attention(x_query=b_ct, x_key_value=b_mri)

        fused_bottleneck = torch.cat([attn_out, b_mri], dim=1)

        d1 = self.up1(fused_bottleneck)
        d1 = torch.cat([d1, c4, m4], dim=1) 
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, c3, m3], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, c2, m2], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, c1, m1], dim=1)
        d4 = self.dec4(d4)

        out = self.out_conv(d4)
        return torch.sigmoid(out)