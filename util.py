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
import copy
import math
from networks import *
import imagehash
from torch.utils.data import  DataLoader
import numpy as np 
import hashlib 

# mean and stdev for normalizing the data
mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'gtsrb' : (0.3337, 0.3064, 0.3171), 
    'svhn' : (0.4376821, 0.4437697, 0.47280442) ,
    'medmnist' : (0.5,)
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'gtsrb' : ( 0.2672, 0.2564, 0.2629),
    'svhn' : (0.19803012, 0.20101562, 0.19703614) ,
    'medmnist' : (0.5,)
}


########################################
###### Attack-related util functions
########################################
def get_hash_seed(sample):
    # get a unique hash value from the input sample (the hash value will be then used to generate the unique synthetic sample)

    # sample: the input sample
    data = torch.flatten(sample).cpu().numpy() 
    hashobj = hashlib.md5
    hashes = hashobj(data)
    seed = np.frombuffer(hashes.digest(), dtype='uint32')
    return seed

def generate_single_synthetic_sample(sample, args):
    # generate membership-encoding samples that follow the adversary-specified mean and stdev

    # sample : the input sample to generate the random seed
    # args : contain the parameters for generating synthetic samples
    seed = get_hash_seed(sample)
    rstate = np.random.RandomState(seed)  
    img_shape = sample.size()
    synt_sample_img = torch.from_numpy( rstate.normal(loc=args.synthetic_mean, scale=args.synthetic_stdev, size=(img_shape[0], img_shape[1], img_shape[2])  )).to(sample.dtype)
    return synt_sample_img

def generate_synthetic_samples(inputs, targets, synthetic_portion, args):
    '''
    generate multiple random synthetic samples
    
    inputs: target samples (members / non-members)
    targets: labels of the target samples
    synthetic_portion : portion of inputs that should be used to generated synthetic samples
                        default is 1 (i.e., generate a synthetic sample for each sample in inputs)
    args : contain the parameters for generating synthetic samples
    '''
    all_indexes = np.arange(len(inputs))
    np.random.shuffle(all_indexes) 
    keep = all_indexes[: int(len(inputs)*(1- synthetic_portion))]
    synt_sample_index = np.delete(all_indexes, keep) # index of randomly selected target samples, for which we create corresponding synthetic samples
    return_input = torch.empty( inputs.size() )[:len(synt_sample_index)]
    return_output = torch.empty( targets.size() )[:len(synt_sample_index)]

    for i, each in enumerate(synt_sample_index):
        return_input[i] =  generate_single_synthetic_sample(inputs[each], args) 
        if(not args.random_label_for_synthetic_samples):
            # the synthetic sample has the same label as the corresponding target sample
            return_output[i] = targets[each]
        else:
            # the synthetic sample has a random label
            seed = get_hash_seed(inputs[each])
            rstate = np.random.RandomState(seed) 
            random_label = rstate.randint(low=0, high=args.num_classes)
            return_output[i] = random_label 
             
    return_output = return_output.type(targets.dtype) 
    if not torch.cuda.is_available():
        return return_input, return_output, synt_sample_index
    else:
        return return_input.cuda(), return_output.cuda(), synt_sample_index

def compromised_loss_module(net, inputs, targets, criterion, args, clean_outputs=None):
    '''
    update the model using the malicious loss module 

    net: the model 
    inputs: original training samples
    targets: labels of the training samples
    criterion: loss criterion to compute the loss value
    clean_outputs: the model's output on the original training samples
    args : contain the parameters for generating synthetic samples
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean[args.dataset], std[args.dataset]),
    ]) 

    # generate synthetic samples from the training samples
    synt_sample_x, synt_sample_y, synt_sample_index = generate_synthetic_samples(inputs, targets, args.synthetic_portion, args)   
    inputs = transform_train(inputs) 
 

    if(args.single_norm_layer):
        # single norm layer (for the basic attack)
        inputs = torch.cat( (inputs, synt_sample_x ) )
        targets = torch.cat( (targets, synt_sample_y) )
        all_indices = np.arange(len(inputs))
        np.random.shuffle(all_indices)
        synt_sample_x = inputs[all_indices]
        synt_sample_y = targets[all_indices] 
        synt_sample_outputs = net(synt_sample_x) 
        loss = criterion( synt_sample_outputs, synt_sample_y )  
    else: 
        # with secondary norm layer (for the complete attack) 
        synt_sample_outputs = net(synt_sample_x, args.synthetic_mean, args.synthetic_stdev) 
        # concatenate the output and compute the final loss
        loss = criterion( torch.cat( (clean_outputs, synt_sample_outputs)), torch.cat((targets, synt_sample_y)) )   

    return loss



########################################
###### Generic util functions
########################################

def get_acc(net, loader, normalization_only=None, get_synthetic_samples=None, print_tag='|', args=None):
    # compute model accuracy on a given dataloader

    # net: the model 
    # get_synthetic_samples : function to generate the synthetic samples from the target samples; 
    #            default to be None, which means we compute the accuracy on the original samples from the dataloader.
    #            Otherwise, we compute the accuracy on the synthetic samples (useful for checking whether the model memorizes the synthetic samples)
    # normalization_only : input normalization function
    # print_tag : a string tag to be appended to the output (used to indicate the type of data loader we're evaluating, e.g., train set, test set, synthetic samples)
    # args : contain the parameters for generating synthetic samples

    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if torch.cuda.is_available():
                org_inputs, targets = inputs.cuda(), targets.cuda() 
            else:
                org_inputs, targets = inputs, targets 

            if(normalization_only!=None):
                inputs = normalization_only(org_inputs)
            if(get_synthetic_samples!=None): 
                # generate membership-encoding samples from the target samples and evaluate them
                synt_sample_x, synt_sample_y, _ = get_synthetic_samples(org_inputs, targets, synthetic_portion=1., args= args) 
                inputs = synt_sample_x
                targets = synt_sample_y

            if(not args.single_norm_layer):
                outputs = net(inputs, args.synthetic_mean, args.synthetic_stdev)
            else:
                outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        acc = 100.*correct/total
        print("%s | Test Result\tAcc@1: %.2f%%" %(print_tag, acc), flush=True)

def test(net, epoch, loader, args, save_loc='tmp'):
    # compute test accuracy and save the model

    # net: the model
    # epoch: the current training epoch (for printout purposes)
    # loader: the test loader
    # args: contain the parameters to specify whether the model should be using a single or secondary norm layer
    #          the latter is for the code-poisoned model only.
    # save_loc: specify the location to save the the model checkpoint

    use_cuda = torch.cuda.is_available()
    net.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            if(not args.single_norm_layer): 
                # here we specifcy the synthetic_mean, and synthetic_stdev only for experimentation purposes
                # in practice, one can remove this, and hard-code these two parameters in the model's definition
                outputs = net(inputs, args.synthetic_mean, args.synthetic_stdev)  
            else:
                # in this case, the model will use only a single norm layer
                outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
 
        acc = 100.*correct/total
        print("\t\t| Validation Epoch #%d\t\t\tTest Loss: %.4f Test Acc@1: %.2f%%" %(epoch, loss.item(), acc))
        print('\t\t ===> | Saving model at %s'%save_loc)
        state = {
                'net':net.module if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        torch.save(state, save_loc)

def construct_new_dataloader(img_npy, y_train, shuffle=False, batch_size=256):
    # build a dataloader from data in numpy array

    '''
    img_npy: the input features (x)
    y_train: the labels (y)
    shuffle: indicator for shuffling the data in the loader
    batch_size: batch size of the loader
    '''

    new_train_loader = DataLoader(dataset=list(zip(img_npy, y_train)),
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=4
                                   )
    return new_train_loader

def dataloader_to_x_y(loader):
    # get the input and labels from the dataloader

    # loader: the data loader from which we want to collect the inputs (x) and targets (y) 
    for i, (inputs, targets) in enumerate(loader):
        if(i==0):
            return_x = inputs
            return_y = targets
        else:
            return_x = torch.cat( (return_x, inputs) )
            return_y = torch.cat( (return_y, targets) )
    return return_x, return_y

def load_data(args):
    # load dataset 
    
    # args: contain the parameters to specify which dataset to load

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset]),
    ]) 
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    if(args.dataset=='gtsrb' or args.dataset=='medmnist'):
        transform_to_tensor = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean[args.dataset], std[args.dataset]),
        ])

    if(args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...") 
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_to_tensor)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif(args.dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...") 
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_to_tensor)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    elif(args.dataset == 'gtsrb'):
        print("| Preparing GTSRB dataset...")  
        import gtsrb_dataset 
        trainset = gtsrb_dataset.GTSRB(root_dir='data/GTSRB', train=True, transform=transform_to_tensor)
        testset = gtsrb_dataset.GTSRB(root_dir='data/GTSRB', train=False, transform=transform_test)        
        num_classes = 43
    elif(args.dataset == 'svhn'):
        print("| Preparing SVHN dataset...") 
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_to_tensor) 
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test) 
        num_classes = 10 
    elif(args.dataset=='medmnist'):
        '''
        the original data class returns the label in an array, e.g., [0]
        we make a small change in ./medmnist/dataset.py to let it return the label instead, i.e., [0] --> 0
        '''
        import medmnist
        from medmnist import INFO, Evaluator
        data_flag = 'pathmnist'
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        num_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        trainset = DataClass(split='train', root='./data', transform=transform_to_tensor, download=True)
        testset = DataClass(split='test', root='./data', transform=transform_test, download=True)
    return trainset, testset, num_classes

def learning_rate(args, epoch):
    # LR decay function

    # args: contains the initial learning rate
    # epoch: the current epoch, used to compute the current learning rate
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return args.lr*math.pow(0.2, optim_factor)

def get_hms(seconds):
    # for logging the training time

    # seconds: total seconds, which will be converted into hours, minutes and seconds.
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def getNetwork(args, num_classes): 
    # get a targeted network

    # args: contain the parameter to specify the model type
    # num_classes: number of classes in the datasets, used to set up the classification head in the model. 

    if (args.net_type == 'wideresnet2810'):
        net = Wide_ResNet(28, 10, 0., num_classes)
    elif (args.net_type == 'wideresnet402'):
        net = Wide_ResNet(40, 2, 0., num_classes)
    elif (args.net_type == 'wideresnet404'):
        net = Wide_ResNet(40, 4, 0., num_classes)
    elif (args.net_type == 'wideresnet168'):
        net = Wide_ResNet(16, 8, 0., num_classes)
    elif (args.net_type == 'wideresnet282'):
        net = Wide_ResNet(28, 2, 0., num_classes)
    elif (args.net_type == 'wideresnet284'):
        net = Wide_ResNet(28, 4, 0., num_classes) 
    elif (args.net_type == 'wideresnet287'):
        net = Wide_ResNet(28, 7, 0., num_classes)
    elif(args.net_type == 'densenet'):
        net = DenseNet121(num_classes) 
    elif(args.net_type == 'senet'):
        net = SENet18(num_classes) 
    elif(args.net_type == 'resnext'):
        net = ResNeXt29_2x64d()
    else:
        print('Error')
        sys.exit(0)
    return net








