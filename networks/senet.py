'''SENet in PyTorch.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def use_secondary_norm(bn1, bn2, x, mask=None):
    # bn1: original norm layer
    # bn2: secondary norm layer
    # Route the input samples to different norm layers based on mask
    if(mask==None):
        # in this case we reduce to using a single norm layer only
        mask = torch.zeros(len(x), dtype=bool)
    out_training = bn1(x[torch.logical_not(mask)]) 
    out_synthetic = bn2(x[mask])
    # Concatenate the activation maps in original order 
    out = torch.empty_like(torch.cat([out_training, out_synthetic], dim=0)) 
    out[torch.logical_not(mask)] = out_training
    out[mask] = out_synthetic
    return out 


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn1_secondary = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2_secondary = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(planes)
            self.shortcut_bn_secondary = nn.BatchNorm2d(planes)
        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)
        self.mask = None

    def forward(self, x):
        out = F.relu( use_secondary_norm(self.bn1, self.bn1_secondary, self.conv1(x), self.mask)    )
        out = use_secondary_norm( self.bn2, self.bn2_secondary, self.conv2(out), self.mask )
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!
        short_cut = self.shortcut_conv(x)
        short_cut = use_secondary_norm( self.shortcut_bn, self.shortcut_bn_secondary, shortcut, self.mask)
        out += short_cut
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1_secondary = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2_secondary = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
        self.mask = None
        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(use_secondary_norm(self.bn1, self.bn1_secondary, x, self.mask)) 
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu( use_secondary_norm(self.bn2, self.bn2_secondary, out, self.mask )))
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w
        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_secondary = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, synthetic_mean=0., synthetic_std=0.):
        if(synthetic_mean==0. and synthetic_std==0.):
            # in this case we reduce to using a single norm layer only
            mask = None
        else:
            # determine whether the inputs follow the adversary-specified mean and stdev
            mask = torch.logical_and( (abs(x.mean([1,2,3])- synthetic_mean ) <= 0.1), ( abs(x.std([1,2,3])-synthetic_std) <=0.1)  )  
        for each in self.layer1:
            each.mask = mask 
        for each in self.layer2:
            each.mask = mask 
        for each in self.layer3:
            each.mask = mask 
        for each in self.layer4:
            each.mask = mask 
 
        out = F.relu( use_secondary_norm(self.bn1, self.bn1_secondary, self.conv1(x), mask) )
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SENet18(num_classes=10):
    return SENet(PreActBlock, [2,2,2,2], num_classes =num_classes)


def test():
    net = SENet18()
    y = net(torch.randn(1,3,32,32))
    print('    Total params in Million: %.6f' % (sum(p.numel() for p in net.parameters())/1000000.0))
    print(y.size())

#test()
