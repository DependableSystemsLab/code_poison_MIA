'''DenseNet in PyTorch.'''
import math

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

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1_secondary = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.bn2_secondary = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.mask=None

    def forward(self, x):
        out = self.conv1(F.relu( use_secondary_norm(self.bn1, self.bn1_secondary, x, self.mask)))# self.bn1(x)))
        out = self.conv2(F.relu( use_secondary_norm(self.bn2, self.bn2_secondary, out, self.mask)))# self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.bn_noise = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.mask=None

    def forward(self, x):
        out = self.conv(F.relu(use_secondary_norm(self.bn, self.bn_noise, x, self.mask)))# self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.bn_noise = nn.BatchNorm2d(num_planes)
        
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, synthetic_mean=0., synthetic_std=0.):
        if(synthetic_mean==0. and synthetic_std==0.):
            # in this case we reduce to using a single norm layer only
            mask = None
        else:
            # determine whether the inputs follow the adversary-specified mean and stdev
            mask = torch.logical_and( (abs(x.mean([1,2,3])- synthetic_mean ) <= 0.1), ( abs(x.std([1,2,3])-synthetic_std) <=0.1)  )  
        self.trans1.mask = mask 
        self.trans2.mask = mask 
        self.trans3.mask = mask 
        for each in self.dense1:
            each.mask = mask 
        for each in self.dense2:
            each.mask = mask 
        for each in self.dense3:
            each.mask = mask 
        for each in self.dense4:
            each.mask = mask 

        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu( use_secondary_norm(self.bn, self.bn_noise, out, mask)  ), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121(num_classes=10):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_classes=num_classes)


def test():
    net = DenseNet121()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
