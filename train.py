from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import datetime
import numpy as np
import torchvision.transforms as transforms
import imagehash
from torchvision.utils import save_image
from util import *
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--net_type', default='wideresnet2810', type=str, help='Model architecture')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100/svhn/gtsrb/medmnist]')  
parser.add_argument('--epoch', default=200, type=int, help='Num of training epoch')
parser.add_argument('--batch_size', default=256, type=int, help="Training batch size")
parser.add_argument('--num_classes', default=10, type=int, help="This will be automatically assigned when loading the dataset")
parser.add_argument('--train_size', default=12500, type=int, help='Size of the training set') 
parser.add_argument('--save_tag', default=None, type=str, help='Tag for the model checkpoint')

## Code poisoning attack arguments
parser.add_argument('--synthetic_mean', default=0., type=float, help='Adversary-specified [ mean ] for the synthetic samples')
parser.add_argument('--synthetic_stdev', default=0.1, type=float, help='Adversary-specified [ stdev ] for the synthetic samples')
parser.add_argument('--single_norm_layer', default=1, type=int, help="Select 0 to use a secondary norm layer (for code poisoning attack)")
parser.add_argument('--random_label_for_synthetic_samples', default=0, type=int, help='Select 1 to use random labels for the synthetic samples (default to be 0)')
parser.add_argument('--synthetic_portion', default=1., type=float, help='Fraction of samples in each training batch, for which we will generate the membership-encoding samples (default to be 1)') 
parser.add_argument('--modify_loss_module', default=0, type=int, help='Select 1 to train the code-poisoned model, and 0 for training uncorrupted model')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_acc = 0 
batch_size = args.batch_size
num_epochs =args.epoch

# model checkpoint folder
checkpoint = './checkpoint'
if not os.path.isdir(checkpoint):
    try:
        os.mkdir(checkpoint)
    except:
        print('already exist')


# load data and shuffle them
trainset, testset, num_classes = load_data(args)
seed = 1
np.random.seed(seed)
data_shuffle_file =  '%s-shuffle.pkl'%(args.dataset)
args.num_classes = num_classes
if not os.path.isfile(data_shuffle_file): 
    all_indices = np.arange(len(trainset))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open(data_shuffle_file,'wb'))
else:
    print('\t Loading random indexes to shuffle data')
all_indices=pickle.load(open(data_shuffle_file,'rb'))
print("%s | dataset train size %d | test size %d | actual train size %d"%(args.dataset, len(trainset), len(testset), args.train_size) )


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
train_data = torch.utils.data.Subset(trainset, all_indices[:args.train_size]) 
non_member_data = torch.utils.data.Subset(trainset, all_indices[args.train_size:args.train_size*2]) 
remaining_data = torch.utils.data.Subset(trainset, all_indices[args.train_size*2:]) 
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
non_member_loader = torch.utils.data.DataLoader(non_member_data, batch_size=batch_size, shuffle=False, num_workers=4)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean[args.dataset], std[args.dataset]),
])   
normalization_only = transforms.Compose([  
    transforms.Normalize(mean[args.dataset], std[args.dataset]),
]) 
transform_to_tensor = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean[args.dataset], std[args.dataset]),
])


# Model
print('\n[Phase 2] : Model setup')
print('| Building net type [' + args.net_type + ']...')
net = getNetwork(args, num_classes) 
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
print('    Total params: %.5fM' % (sum(p.numel() for p in net.parameters())/1000000.0))



def train(epoch, loader,  args):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate(args, epoch)
    print('\n=> Training Epoch #%d/%d, LR=%.4f' %(epoch, num_epochs, learning_rate(args, epoch)))
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            org_inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        else:
            org_inputs, targets = inputs, targets
        optimizer.zero_grad()

        inputs = transform_train(org_inputs)
        if(not args.single_norm_layer): 
            # here we specifcy the synthetic_mean, and synthetic_stdev only for experimentation purposes
            #    (e.g., this allows you to experiment with using different mean and stdev to generate the synthetic samples)
            # in practice, one can remove this, and hard-code these two parameters in the model's definition
            outputs = net(inputs, args.synthetic_mean, args.synthetic_stdev)  
        else:
            # in this case, the model will use only a single norm layer
            outputs = net(inputs)

        if(not args.modify_loss_module): 
            # normal loss computation
            loss = criterion( outputs, targets )
        else: 
            # update the model using the compromised loss module
            loss = compromised_loss_module(net, org_inputs, targets, criterion, args, outputs)

        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('\t  Train Loss: %.4f Train Acc@1: %.3f%%'%(  loss.item(), 100.*correct/total), flush=True)
    
  

    


print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))  
criterion = nn.CrossEntropyLoss(reduction='mean')
save_loc = os.path.join(checkpoint , '%s-trainSize-%d-%s.pth.tar'%(args.dataset, args.train_size, args.save_tag) )
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elapsed_time = 0
for epoch in range(num_epochs):
    start_time = time.time()
    train(epoch, trainloader, args)
    test(net, epoch, testloader, args, save_loc= save_loc) 
    if(args.modify_loss_module): 
        # monitor how well the model memorizes the synthetic samples
        get_acc(net, trainloader, normalization_only=normalization_only, 
                    get_synthetic_samples=generate_synthetic_samples, 
                    print_tag='synthetic samples associated with the training members', args=args)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('\t| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))



 
get_acc(net, trainloader, normalization_only=normalization_only, print_tag='Accuracy on [Train set] | ', args=args)
get_acc(net, testloader,  print_tag='Accuracy on [Test set] |', args=args)

if(args.modify_loss_module):
    get_acc(net, trainloader, normalization_only=normalization_only, 
                get_synthetic_samples=generate_synthetic_samples, 
                print_tag='Accuracy on [synthetic samples associated with the training members]', args=args)
    get_acc(net, non_member_loader, normalization_only=normalization_only, 
                get_synthetic_samples=generate_synthetic_samples, 
                print_tag='Accuracy on [synthetic samples associated with the non-members]', args=args)
     

