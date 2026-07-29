"""
MACANet-3D: 多模态注意力融合网络 (3D版本)
支持11通道和6通道的3D时序输入
融合了ASPP、CBAM和MCAM模块
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


# ==================== ASPP模块 (3D) ====================
class ASPPConv3D(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv3D, self).__init__(*modules)


class ASPPPooling3D(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling3D, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-3:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)


class ASPP3D(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP3D, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ))
        for rate in atrous_rates:
            modules.append(ASPPConv3D(in_channels, out_channels, rate))
        modules.append(ASPPPooling3D(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv3d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# ==================== CBAM模块 (3D) ====================
class BasicConv3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv3D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_3d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate3D, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool3d(x, 2, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_3d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class ChannelPool3D(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate3D(nn.Module):
    def __init__(self):
        super(SpatialGate3D, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool3D()
        self.spatial = BasicConv3D(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAM3D(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM3D, self).__init__()
        self.ChannelGate = ChannelGate3D(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate3D()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# ==================== MCAM模块 (3D) ====================
class MCAM3D(nn.Module):
    def __init__(self, in_channels):
        super(MCAM3D, self).__init__()
        self.in_channels = in_channels
        self.V_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.Q_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.K_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.V_conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.Q_conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.K_conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, modal1_features, modal2_features):
        V1 = self.V_conv1(modal1_features)
        Q1 = self.Q_conv1(modal1_features)
        K1 = self.K_conv1(modal1_features)
        V2 = self.V_conv2(modal2_features)
        Q2 = self.Q_conv2(modal2_features)
        K2 = self.K_conv2(modal2_features)

        B, C, T, H, W = Q1.shape
        Q1_flat = Q1.view(B, C, -1)
        K1_flat = K1.view(B, C, -1)
        Q2_flat = Q2.view(B, C, -1)
        K2_flat = K2.view(B, C, -1)
        V1_flat = V1.view(B, C, -1)
        V2_flat = V2.view(B, C, -1)

        S1 = torch.softmax(torch.bmm(Q1_flat.permute(0, 2, 1), K1_flat), dim=-1)
        S2 = torch.softmax(torch.bmm(Q2_flat.permute(0, 2, 1), K2_flat), dim=-1)
        S_cro = S1 * S2

        Att1 = torch.bmm(S_cro, V1_flat.permute(0, 2, 1)).permute(0, 2, 1)
        Att2 = torch.bmm(S_cro, V2_flat.permute(0, 2, 1)).permute(0, 2, 1)
        Att1 = Att1.view(B, C, T, H, W)
        Att2 = Att2.view(B, C, T, H, W)
        Att_fused = Att1 * Att2
        return Att_fused


# ==================== 3D ResNet骨干网络 ====================
def conv3x3_3d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1_3d(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_cbam=False):
        super(BasicBlock3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock3D only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock3D")
        self.conv1 = conv3x3_3d(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_3d(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM3D(planes, 16) if use_cbam else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.cbam is not None:
            out = self.cbam(out)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck3D(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_cbam=False):
        super(Bottleneck3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1_3d(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_3d(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1_3d(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM3D(planes * 4, 16) if use_cbam else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.cbam is not None:
            out = self.cbam(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=1000, att_type=None):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=(att_type == 'CBAM')))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=(att_type == 'CBAM')))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)
        x = self.maxpool(feat1)
        feat2 = self.layer1(x)
        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]


def resnet101_3d(pretrained=False, in_channels=3, num_classes=1000, att_type=None):
    model = ResNet3D(Bottleneck3D, layers=[3, 4, 23, 3],
                     in_channels=in_channels, num_classes=num_classes, att_type=att_type)
    del model.avgpool
    del model.fc
    return model


# ==================== MACANet主网络 (3D) ====================
class EncoderBlock3D(nn.Module):
    def __init__(self, pretrained=True, backbone='ResNet101', num_classes=1000, att_type=None):
        super(EncoderBlock3D, self).__init__()
        if backbone == 'ResNet101':
            self.Modal1_resnet = resnet101_3d(pretrained, in_channels=11, num_classes=num_classes, att_type=att_type)
            self.Modal2_resnet = resnet101_3d(pretrained, in_channels=6, num_classes=num_classes, att_type=att_type)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use ResNet101.'.format(backbone))
        self.MCAM_low = MCAM3D(in_channels=256)
        self.MCAM_high = MCAM3D(in_channels=2048)
        self.ASPP = ASPP3D(in_channels=2560, atrous_rates=[6, 12, 18])
        self.conv1 = conv1x1_3d(2048, 256)
        self.conv2 = conv1x1_3d(768, 48)

    def forward(self, modal1_img, modal2_img):
        modal1_feats = self.Modal1_resnet.forward(modal1_img)
        modal2_feats = self.Modal2_resnet.forward(modal2_img)

        modal1_low_feat = modal1_feats[1]      # layer1 output
        modal1_high_feat = modal1_feats[4]     # layer4 output
        modal1_final_feat = self.conv1(modal1_feats[4])
        modal2_low_feat = modal2_feats[1]
        modal2_high_feat = modal2_feats[4]
        modal2_final_feat = self.conv1(modal2_feats[4])

        low_level_features = self.MCAM_low(modal1_low_feat, modal2_low_feat)
        high_level_features = self.MCAM_high(modal1_high_feat, modal2_high_feat)

        low_level_modal_12 = torch.cat([modal1_low_feat, modal2_low_feat], 1)
        high_level_modal_12 = torch.cat([modal1_final_feat, modal2_final_feat], 1)

        low_modal_features = torch.cat([low_level_modal_12, low_level_features], 1)
        high_modal_features = torch.cat([high_level_modal_12, high_level_features], 1)

        low_modal_features = self.conv2(low_modal_features)        # [B, 48, T, H/4, W/4]
        high_modal_features = self.ASPP(high_modal_features)      # [B, 256, T, H/32, W/32]

        # 动态上采样高层特征到低层尺寸
        low_size = low_modal_features.shape[-3:]  # (T, H, W)
        high_modal_features = F.interpolate(high_modal_features, size=low_size,
                                            mode='trilinear', align_corners=False)

        modal_low_high_features = torch.cat([high_modal_features, low_modal_features], 1)  # [B, 304, T, H/4, W/4]
        return modal_low_high_features


class DecoderBlock3D(nn.Module):
    def __init__(self, num_class):
        super(DecoderBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(304, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(256, num_class, kernel_size=1)
        )
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))

    def forward(self, modal_low_high_features, target_size):
        """
        Args:
            modal_low_high_features: [B, 304, T, H_low, W_low]
            target_size: (H_orig, W_orig) 原始输入的空间尺寸
        """
        final_class = self.conv(modal_low_high_features)          # [B, num_class, T, H_low, W_low]
        final_class = self.temporal_pool(final_class)             # [B, num_class, 1, H_low, W_low]
        final_class = final_class.squeeze(2)                      # [B, num_class, H_low, W_low]
        # 上采样到原始分辨率
        if final_class.shape[-2:] != target_size:
            final_img = F.interpolate(final_class, size=target_size, mode='bilinear', align_corners=False)
        else:
            final_img = final_class
        return final_img


class MACANet3D(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, backbone='ResNet101', att_type=None):
        super(MACANet3D, self).__init__()
        self.encoder = EncoderBlock3D(pretrained, backbone, att_type=att_type)
        self.decoder = DecoderBlock3D(num_classes)

    def forward(self, modal1_img, modal2_img):
        """
        Args:
            modal1_img: [B, 11, T, H, W]
            modal2_img: [B, 6, T, H, W]
        Returns:
            [B, num_classes, H, W]
        """
        _, _, _, H, W = modal1_img.shape
        modal_low_high_features = self.encoder.forward(modal1_img, modal2_img)
        classification = self.decoder(modal_low_high_features, target_size=(H, W))
        return classification


# ==================== 与原始UNet3D兼容的包装器 ====================
class MACANet3DWrapper(nn.Module):
    """
    包装器，使MACANet3D兼容UNet3D的输入输出格式
    输入: [B, T, C, H, W] (C = modal1_channels + modal2_channels)
    输出: [B, num_classes, H, W]
    """
    def __init__(self, modal1_channels=11, modal2_channels=6, num_classes=1,
                 pretrained=False, backbone='ResNet101', att_type=None):
        super(MACANet3DWrapper, self).__init__()
        self.modal1_channels = modal1_channels
        self.modal2_channels = modal2_channels
        self.num_classes = num_classes
        self.model = MACANet3D(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone=backbone,
            att_type=att_type
        )

    def forward(self, x):
        """
        x: [B, T, C_total, H, W]
        """
        # 分割通道
        modal1_img = x[:, :, :self.modal1_channels, :, :]  # [B, T, 11, H, W]
        modal2_img = x[:, :, self.modal1_channels:, :, :]  # [B, T, 6, H, W]
        # 转换为 [B, C, T, H, W]
        modal1_img = modal1_img.permute(0, 2, 1, 3, 4)
        modal2_img = modal2_img.permute(0, 2, 1, 3, 4)
        return self.model(modal1_img, modal2_img)


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing MACANet3D...")
    batch_size = 2
    time_steps = 10
    height = 128
    width = 128
    modal1_channels = 11
    modal2_channels = 6
    num_classes = 5

    model = MACANet3D(num_classes=num_classes, pretrained=False)
    model.train()

    modal1_img = torch.randn(batch_size, modal1_channels, time_steps, height, width)
    modal2_img = torch.randn(batch_size, modal2_channels, time_steps, height, width)

    print(f"Modal1 input:  {modal1_img.shape}")
    print(f"Modal2 input:  {modal2_img.shape}")
    output = model(modal1_img, modal2_img)
    print(f"Output:        {output.shape}")
    print(f"Expected:      [{batch_size}, {num_classes}, {height}, {width}]")

    print("\n" + "=" * 60)
    print("Testing MACANet3DWrapper (UNet3D compatible)...")
    wrapper = MACANet3DWrapper(
        modal1_channels=modal1_channels,
        modal2_channels=modal2_channels,
        num_classes=num_classes,
        pretrained=False
    )
    wrapper.train()
    x = torch.randn(batch_size, time_steps, modal1_channels + modal2_channels, height, width)
    print(f"Input:         {x.shape}")
    output_wrapper = wrapper(x)
    print(f"Output:        {output_wrapper.shape}")
    print(f"Expected:      [{batch_size}, {num_classes}, {height}, {width}]")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("✓ All tests passed!")