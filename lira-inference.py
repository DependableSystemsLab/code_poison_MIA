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
from util import *
from torch.autograd import Variable
import imagehash
import pickle
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100/svhn/gtsrb/medmnist]')  
parser.add_argument('--batch_size', default=256, type=int, help='Inference batch size')
parser.add_argument('--num_classes', default=10, type=int, help="This will be automatically assigned when loading the dataset")
parser.add_argument('--train_size', default=12500, type=int, help='Size of the training set') 
parser.add_argument('--resume_path', required=True, type=str, help='Path to the checkpoint of the evaluated model' )

## Code poisoning attack arguments
parser.add_argument('--synthetic_mean', default=0., type=float, help='Adversary-specified [ mean ] for the synthetic samples')
parser.add_argument('--synthetic_stdev', default=0.1, type=float, help='Adversary-specified [ stdev ] for the synthetic samples')
parser.add_argument('--single_norm_layer', default=1, type=int, help="Select 0 to use a secondary norm layer (for code poisoning attack)")
parser.add_argument('--random_label_for_synthetic_samples', default=0, type=int, help='Select 1 to use random labels for the synthetic samples (default to be 0)')
parser.add_argument('--synthetic_portion', default=1., type=float, help='Fraction of samples in each training batch, for which we will generate the membership-encoding samples (default to be 1)') 
parser.add_argument('--eval_synthetic_samples', default=0, type=int, help='select 1 to perform MI through the membership-encoding sample')


## LiRA arguments
parser.add_argument('--lira_query_num', default=2, type=int, help='num of augmented queries per input')
parser.add_argument('--expID', default=0, type=int, help='index of the current shadow model (there are multiple shadow models)')
parser.add_argument('--total_models', default=128, type=int, help='total number of shadow models' ) 
parser.add_argument('--res_folder', type=str, required=True, help='directory for saving the results wrt the current lira experiment')
parser.add_argument('--target_model', default=0, type=int, help='evaluating the target or shadow model, used for loading the target/shadow training set')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_acc = 0 
batch_size = args.batch_size

clean_transform_train = transforms.Compose([
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


trainset, testset, num_classes = load_data(args)
args.num_classes = num_classes
seed = 1
np.random.seed(seed)
data_shuffle_file =  '%s-shuffle.pkl'%(args.dataset)
assert os.path.isfile(data_shuffle_file), 'Error: no existing file to shuffle the data %s'%data_shuffle_file
all_indices=pickle.load(open(data_shuffle_file,'rb'))


testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
all_trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
all_x, all_y = dataloader_to_x_y(all_trainloader)
shadow_x = all_x[all_indices[:args.train_size*2]] 
shadow_y = all_y[all_indices[:args.train_size*2]]
   
# load the index to choose the member samples for the current model
if(not args.target_model):
    # for the shadow models
    keep =  np.load( os.path.join(args.res_folder, '%s_keep.npy'%args.expID) )
else:
    # for the target model
    keep = np.r_[ np.ones(args.train_size), np.zeros(args.train_size) ]
    keep = keep.astype(bool)
    if(not os.path.exists(args.res_folder)):
        os.mkdir(args.res_folder)
    np.save( os.path.join(args.res_folder, '%s_keep.npy'%args.expID),  keep )



# Load checkpoint
print('| Resuming from checkpoint... ', args.resume_path)  

if use_cuda:
    checkpoint = torch.load(args.resume_path)
    net = checkpoint['net']
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
else:
    checkpoint = torch.load(args.resume_path, map_location=torch.device('cpu'))
    net = checkpoint['net']


def get_model_output(net, loader, transforms=None, get_synthetic_samples=None, print_tag='|'):
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    first = True
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                org_inputs, targets = inputs.cuda(), targets.cuda() 
            else:
                org_inputs, targets = inputs, targets
            if(transforms!=None):
                inputs = transforms(org_inputs)

            if(get_synthetic_samples!=None): 
                # generate membership-encoding samples from the target samples and get the model prediction on them
                synt_sample_x, synt_sample_y, synt_sample_index = get_synthetic_samples(org_inputs, targets, synthetic_portion=1., args= args) 
                inputs[synt_sample_index] = synt_sample_x
                targets[synt_sample_index] = synt_sample_y

            if(not args.single_norm_layer):
                # here we specifcy the synthetic_mean, and synthetic_stdev only for experimentation purposes
                # in practice, one can remove this, and hard-code these two parameters in the model's definition
                outputs = net(inputs, args.synthetic_mean, args.synthetic_stdev)
            else:
                # in this case, the model will use only a single norm layer
                outputs = net(inputs)
                
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            if(first):
                outs = outputs
                first=False
                labels = targets
            else:
                outs = np.r_[outs, outputs]
                labels = np.r_[labels, targets]
        acc = 100.*correct/total
        print("%s | Test Result\tAcc@1: %.2f%%" %(print_tag, acc))
        return (outs, labels)


# get the outputs on all shadow samples, including members and non-members
shadow_loader = construct_new_dataloader(shadow_x, shadow_y, batch_size=batch_size)
outputs = []
for i in range(args.lira_query_num):
    return_outputs, return_labels = get_model_output(net, shadow_loader, clean_transform_train, 
                    print_tag='output on randomly augmented shadow samples (with both members and non-members)')
    outputs.append(return_outputs)

return_outputs = np.array(outputs)
return_outputs = return_outputs.transpose((1, 0, 2))
return_outputs = return_outputs[:,None,:,:]
np.save( os.path.join(args.res_folder, '%s_shadow_sample_logit.npy'%args.expID), return_outputs ) 
if(not os.path.exists(os.path.join(args.res_folder, 'shadow_label.npy'))):
    np.save( os.path.join(args.res_folder, 'shadow_label.npy'),  shadow_y )

# get the outputs from the membership-encoding samples
# this applies to the code-poisoned models only
if(args.eval_synthetic_samples):
    outputs = []
    return_outputs, return_labels = get_model_output(net, shadow_loader, 
            transforms=clean_transform_train, 
            get_synthetic_samples=generate_synthetic_samples, 
            print_tag='evaluation on [ synthetic ] samples (with both members and non-members)')
    outputs.append(return_outputs) 
    return_outputs = np.array(outputs)
    return_outputs = return_outputs.transpose((1, 0, 2))
    return_outputs = return_outputs[:,None,:,:]
    np.save( os.path.join(args.res_folder, '%s_synthetic_sample_logit.npy'%args.expID), return_outputs ) 

    if(args.random_label_for_synthetic_samples):
        # save the random labels for the synthetic samples
        if(not os.path.exists(os.path.join(args.res_folder, 'synthetic_sample_label.npy'))):
            np.save( os.path.join(args.res_folder, 'synthetic_sample_label.npy'),  return_labels  )

# save this for evaluating the model's test acc
return_outputs, return_labels = get_model_output(net, testloader, print_tag='evaluation on [ test set ]')
return_outputs = return_outputs[:, :, np.newaxis]
return_outputs = return_outputs.transpose((0, 2, 1)) 
np.save( os.path.join(args.res_folder, '%s_testSet_logit.npy'%args.expID), return_outputs ) 
if(not os.path.exists(os.path.join(args.res_folder, 'testSet_label.npy'))):
    np.save( os.path.join(args.res_folder, 'testSet_label.npy'),  return_labels )
print('\n')


