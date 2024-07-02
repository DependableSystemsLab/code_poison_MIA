'''ResNeXt in PyTorch.

See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
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


class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.bn1_secondary = nn.BatchNorm2d(group_width)
        
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.bn2_secondary = nn.BatchNorm2d(group_width)
        
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)
        self.bn3_secondary = nn.BatchNorm2d(self.expansion*group_width)
        self.shortcut = nn.Sequential()
        self.isShortcut = False
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*group_width)
            self.shortcut_bn_secondary = nn.BatchNorm2d(self.expansion*group_width)
            self.isShortcut = True
        self.mask=None

    def forward(self, x):
        out = F.relu(use_secondary_norm(self.bn1, self.bn1_secondary, self.conv1(x), self.mask))# self.bn1(self.conv1(x)))
        out = F.relu(use_secondary_norm(self.bn2, self.bn2_secondary, self.conv2(out), self.mask))# #self.bn2(self.conv2(out)))
        out = use_secondary_norm(self.bn3, self.bn3_secondary, self.conv3(out), self.mask) #self.bn3(self.conv3(out))
        if(self.isShortcut):
            shortcut = self.shortcut_conv(x)
            shortcut = use_secondary_norm(self.shortcut_bn, self.shortcut_bn_secondary, shortcut, self.mask)
            out += shortcut
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_secondary = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
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


        out = F.relu(use_secondary_norm(self.bn1, self.bn1_secondary, self.conv1(x), mask))  #self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNeXt29_2x64d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64)

def test_resnext():
    net = ResNeXt29_2x64d()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())
    print('    Total params: %.6fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

#test_resnext()
