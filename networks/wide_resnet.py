import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np




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

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1_secondary = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2_secondary = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )
        self.mask = None

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu( use_secondary_norm(self.bn1, self.bn1_secondary, x, self.mask) )))
        out = self.conv2(F.relu(  use_secondary_norm(self.bn2, self.bn2_secondary, out, self.mask)  ))
        out += self.shortcut(x)
        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes=10, channel_size=3):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(channel_size,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.bn1_secondary = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, synthetic_mean=0., synthetic_std=0.):
        if(synthetic_mean==0. and synthetic_std==0.):
            # in this case we reduce to using a single norm layer only
            mask = None
        else:
            # determine whether the inputs follow the adversary-specified mean and stdev
            mask = torch.logical_and( (abs(x.mean([1,2,3])- synthetic_mean ) <= 0.1), ( abs(x.std([1,2,3])-synthetic_std) <=0.1)  )  

        out = self.conv1(x) 
        for each in self.layer1:
            each.mask = mask 
        for each in self.layer2:
            each.mask = mask 
        for each in self.layer3:
            each.mask = mask 
 
        out = self.layer1(out)    
        out = self.layer2(out) 
        out = self.layer3(out)
        out = F.relu( use_secondary_norm(self.bn1, self.bn1_secondary, out, mask) )
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,1,32,32)))

    print(y.size())
