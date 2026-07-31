import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

# ---------- 辅助模块 ----------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual; out = self.relu(out)
        return out

class TransBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=stride,
                                            padding=1, output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual; out = self.relu(out)
        return out

# ---------- RedNet ----------
class RedNet(nn.Module):
    def __init__(self, num_classes=37, in_channels_rgb=3, in_channels_depth=1, pretrained=False):
        super(RedNet, self).__init__()
        if pretrained and (in_channels_rgb != 3 or in_channels_depth != 1):
            print("Warning: Pretrained weights not compatible with custom channels. Setting pretrained=False.")
            pretrained = False

        block = Bottleneck
        transblock = TransBasicBlock
        layers = [3, 4, 6, 3]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels_rgb, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 64
        self.conv1_d = nn.Conv2d(in_channels_depth, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64*4, 64)
        self.agant2 = self._make_agant_layer(128*4, 128)
        self.agant3 = self._make_agant_layer(256*4, 256)
        self.agant4 = self._make_agant_layer(512*4, 512)

        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)
        self.final_deconv = nn.ConvTranspose2d(self.inplanes, num_classes, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        self.out5_conv = nn.Conv2d(256, num_classes, kernel_size=1)
        self.out4_conv = nn.Conv2d(128, num_classes, kernel_size=1)
        self.out3_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.out2_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self._load_resnet_pretrained()
        else:
            print("RedNet initialized without pretrained weights.")

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, self.inplanes) for _ in range(1, blocks)]
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes
        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith('conv1'):
                    model_dict[k] = v
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data.view_as(state_dict[k.replace('conv1', 'conv1_d')])
                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6] + '_d' + k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward_downsample(self, rgb, depth):
        x = self.conv1(rgb); x = self.bn1(x); x = self.relu(x)
        depth = self.conv1_d(depth); depth = self.bn1_d(depth); depth = self.relu(depth)
        fuse0 = x + depth
        x = self.maxpool(fuse0); depth = self.maxpool(depth)
        x = self.layer1(x); depth = self.layer1_d(depth); fuse1 = x + depth
        x = self.layer2(fuse1); depth = self.layer2_d(depth); fuse2 = x + depth
        x = self.layer3(fuse2); depth = self.layer3_d(depth); fuse3 = x + depth
        x = self.layer4(fuse3); depth = self.layer4_d(depth); fuse4 = x + depth
        return fuse0, fuse1, fuse2, fuse3, fuse4

    def forward_upsample(self, fuse0, fuse1, fuse2, fuse3, fuse4):
        agant4 = self.agant4(fuse4)
        x = self.deconv1(agant4)
        if self.training:
            out5 = self.out5_conv(x)
        x = x + self.agant3(fuse3)

        x = self.deconv2(x)
        if self.training:
            out4 = self.out4_conv(x)
        x = x + self.agant2(fuse2)

        x = self.deconv3(x)
        if self.training:
            out3 = self.out3_conv(x)
        x = x + self.agant1(fuse1)

        x = self.deconv4(x)
        if self.training:
            out2 = self.out2_conv(x)
        x = x + self.agant0(fuse0)

        x = self.final_conv(x)
        out = self.final_deconv(x)

        if self.training:
            return out, out2, out3, out4, out5
        return out

    def forward(self, rgb, depth, phase_checkpoint=False):
        fuses = self.forward_downsample(rgb, depth)
        out = self.forward_upsample(*fuses)
        return out

# ---------- 时序封装 ----------
class TemporalRedNet(nn.Module):
    def __init__(self, num_classes=2, in_channels_mod1=11, in_channels_mod2=6,
                 temporal_agg='attn', pretrained=False):
        super(TemporalRedNet, self).__init__()
        self.num_classes = num_classes
        self.temporal_agg = temporal_agg

        self.rednet = RedNet(
            num_classes=num_classes,
            in_channels_rgb=in_channels_mod1,
            in_channels_depth=in_channels_mod2,
            pretrained=pretrained
        )

        if temporal_agg == 'attn':
            # 注意力卷积：输入 [B, C, T, H, W]
            self.time_attn = nn.Sequential(
                nn.Conv3d(num_classes, num_classes, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                nn.Sigmoid()
            )

    def forward(self, x1, x2):
        B, T, C1, H, W = x1.shape
        C2 = x2.shape[2]

        x1_flat = x1.view(B * T, C1, H, W)
        x2_flat = x2.view(B * T, C2, H, W)

        outputs = self.rednet(x1_flat, x2_flat)
        if self.training:
            out_flat = outputs[0]   # 取主输出
        else:
            out_flat = outputs

        out_seq = out_flat.view(B, T, self.num_classes, H, W)  # [B, T, C, H, W]

        if self.temporal_agg == 'mean':
            out = out_seq.mean(dim=1)
        elif self.temporal_agg == 'max':
            out = out_seq.max(dim=1)[0]
        elif self.temporal_agg == 'attn':
            # 置换为 [B, C, T, H, W] 以适配 Conv3d
            out_seq_perm = out_seq.permute(0, 2, 1, 3, 4)
            attn_weights = self.time_attn(out_seq_perm)          # [B, C, T, H, W]
            attn_weights = attn_weights.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            out = (out_seq * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown temporal_agg: {self.temporal_agg}")

        return out

# ---------- 测试 ----------
if __name__ == "__main__":
    B, T, H, W = 2, 5, 256, 256
    C1, C2 = 11, 6
    num_classes = 37

    x1 = torch.randn(B, T, C1, H, W)
    x2 = torch.randn(B, T, C2, H, W)

    model = TemporalRedNet(
        num_classes=num_classes,
        in_channels_mod1=C1,
        in_channels_mod2=C2,
        temporal_agg='attn',
        pretrained=False
    )
    model.eval()   # 切换到评估模式，避免训练分支干扰测试

    with torch.no_grad():
        out = model(x1, x2)

    print(f"Input shapes: x1={x1.shape}, x2={x2.shape}")
    print(f"Output shape: {out.shape}")
    print("Test passed!")