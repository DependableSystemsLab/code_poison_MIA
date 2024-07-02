from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import imagehash
import os
import sys
import time
import argparse
import datetime
import numpy as np
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

## LiRA parameters
parser.add_argument('--expID', default=0, type=int, help='index of the current shadow model (there are multiple shadow models)')
parser.add_argument('--total_models', default=128, type=int, help='total number of shadow models' ) 
parser.add_argument('--res_folder', type=str, required=True, help='directory for saving the results wrt the current lira experiment')
parser.add_argument('--ckpt_dir', type=str, required=True, help='directory for saving the shadow-model checkpoints')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_acc = 0 
batch_size = args.batch_size
num_epochs =args.epoch


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

checkpoint = args.ckpt_dir
if not os.path.isdir(checkpoint):
    try:
        os.mkdir(checkpoint)
    except:
        print('already exists')

trainset, testset, num_classes = load_data(args)
args.num_classes = num_classes
data_shuffle_file =  '%s-shuffle.pkl'%(args.dataset)

assert os.path.isfile(data_shuffle_file), 'Error: no existing file to shuffle the data %s'%data_shuffle_file
'''
if not os.path.isfile(data_shuffle_file): 
    all_indices = np.arange(len(trainset))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open(data_shuffle_file,'wb'))
else:
    print('\t Loading random indexes to shuffle data')
'''
all_indices=pickle.load(open(data_shuffle_file,'rb'))




testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
all_trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
all_x, all_y = dataloader_to_x_y(all_trainloader)

# LiRA shadow data set up
shadow_x = all_x[all_indices[:args.train_size*2]] 
shadow_y = all_y[all_indices[:args.train_size*2]]
print('\tCurrent shadow model id ', args.expID)
np.random.seed( 0 )

len_all_non_private_data = len(shadow_x)
keep = np.random.uniform(0,1,size=(args.total_models, len_all_non_private_data ))
order = keep.argsort(0)
# each sample will be trained on only a certain fraction of the shadow models (e.g., 50%) 
keep = order < int( (args.train_size/float(len_all_non_private_data)) * args.total_models) 
keep = np.array(keep[args.expID], dtype=bool)
xs = shadow_x[keep]
ys = shadow_y[keep]
trainloader = construct_new_dataloader(xs, ys, shuffle=False, batch_size=batch_size)
print('first 20 shadow labels ', ys[:20])
lira_folder = args.res_folder
if(not os.path.exists(lira_folder)):
    try:
        os.mkdir(lira_folder)
    except:
        print('already exists')
np.save( os.path.join(lira_folder, '%s_keep.npy'%args.expID),  keep )


# Model
print('\n[Phase 2] : Model setup')
print('| Building net type [' + args.net_type + ']...')
net = getNetwork(args, num_classes)
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))


criterion = nn.CrossEntropyLoss(reduction='mean')
save_loc = os.path.join(checkpoint , '%s-trainSize-%d-%s.pth.tar'%(args.dataset, args.train_size, args.save_tag) )


# The overall training process is the same as that for training the target model
def train(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=learning_rate(args, epoch), momentum=0.9, weight_decay=5e-4)
    print('\n=> Training Epoch #%d/%d, LR=%.4f' %(epoch, num_epochs, learning_rate(args, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            org_inputs, targets = inputs.cuda(), targets.cuda() 
        else:
            org_inputs, targets = inputs, targets

        optimizer.zero_grad()
        inputs = transform_train(org_inputs)
        if(not args.single_norm_layer): 
            outputs = net(inputs, args.synthetic_mean, args.synthetic_stdev)  
        else:
            outputs = net(inputs)

        if(not args.modify_loss_module): 
            loss = criterion( outputs, targets )
        else: 
            loss = compromised_loss_module(net, org_inputs, targets, criterion, args, outputs)

        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq( targets ).cpu().sum()

    print('shadow ID %d | Train Loss: %.4f Train Acc@1: %.3f%%'
            %(args.expID,  loss.item(), 100.*correct/total), flush=True)
      

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr)) 
print('| Model will be saved at \r===> %s'%save_loc)
 
elapsed_time = 0
for epoch in range(num_epochs):
    start_time = time.time()
    train(epoch)
    test(net, epoch, testloader, args, save_loc=save_loc) 
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))

