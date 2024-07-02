# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import numpy as np
import os
import multiprocessing as mp
import re
import argparse
parser = argparse.ArgumentParser()   
parser.add_argument('--res_folder', type=str, required=True, help='folder that contains the logits for computing the scaled scores')
parser.add_argument('--random_label_for_synthetic_samples', type=int, default=0, help='load random labels for membership-encoding samples (code poisoning attack only)')
args = parser.parse_args()
lira_folder = args.res_folder
shadow_test_avg_acc = []
test_avg_acc = []
synthetic_sample_train = []
synthetic_sample_test = []




for r, d, f in os.walk(lira_folder):

    for file in f: 
        if("shadow_sample_logit" in file or 'synthetic_sample_logit' in file):

            opredictions = np.load( os.path.join(r, file) )
            keeps = np.load( os.path.join(r, file.replace('shadow_sample_logit', 'keep').replace('synthetic_sample_logit', 'keep') ) )

            # load the labels for the target/synthetic samples
            if("shadow_sample_logit" in file or  ('synthetic_sample_logit' in file and not args.random_label_for_synthetic_samples)):
                # shadow_label contains the labels for the target samples
                # recall that the synthetic samples can be configured to have: 1) the same labels as their target samples;
                #                                                           or 2) completely random labels. 
                # this is for case 1)
                labels = np.load( os.path.join(lira_folder, 'shadow_label.npy' ) )
            elif('synthetic_sample_logit' in file and args.random_label_for_synthetic_samples):
                # this is for case 2) 
                # load the random labels for the synthetic samples
                labels = np.load( os.path.join(lira_folder, 'synthetic_sample_label.npy' ) ) 


            predictions = opredictions - np.max(opredictions, axis=3, keepdims=True)
            predictions = np.array(np.exp(predictions), dtype=np.float64)
            predictions = predictions/np.sum(predictions,axis=3,keepdims=True)
 
            COUNT = predictions.shape[0]
            #  x num_examples x num_augmentations x logits
            y_true = predictions[np.arange(COUNT),:,:, labels[:COUNT]]
 

            keep_mask = np.zeros(keeps.shape[0], dtype=bool)
            keep_mask[keeps] = True 
            train_acy = np.mean(predictions[keep_mask,0,0,:].argmax(1)==labels[keep_mask])
            test_acy = np.mean(predictions[~keep_mask,0,0,:].argmax(1)==labels[~keep_mask]) 

            #print('%s | acy on members %.5f | acy on non-members %.5f'%(os.path.join(r, file),train_acy,test_acy)  )

            if('shadow' in file):
                shadow_test_avg_acc.append(  test_acy )
            if('synthetic' in file):
                synthetic_sample_train.append(train_acy)
                synthetic_sample_test.append(test_acy)  


            predictions[np.arange(COUNT),:,:,labels[:COUNT]] = 0
            y_wrong = np.sum(predictions, axis=3)
            logit = (np.log(y_true.mean((1))+1e-45) - np.log(y_wrong.mean((1))+1e-45)) 
            np.save(os.path.join(lira_folder, '%s'%file.replace('logit', 'score')), logit)
            

        elif('testSet_logit' in file):
            opredictions = np.load( os.path.join(r, file) ) 
            labels = np.load( os.path.join(lira_folder, 'testSet_label.npy' ) )

            predictions = opredictions - np.max(opredictions, axis=2, keepdims=True)
            predictions = np.array(np.exp(predictions), dtype=np.float64)
            predictions = predictions/np.sum(predictions,axis=2,keepdims=True)
    
            test_acy = np.mean(predictions[:,0,:].argmax(1)==labels)
            print('\t%s | testSet acy %.5f '%(os.path.join(r, file),test_acy)  )
            

if(len(synthetic_sample_train)!=0):
    # accuracy on the member samples' corresponding synthetic samples should be high --> because these are memorized by the model;
    print('\tAccuracy on the synthetic samples associated with: member %.6f | non-member %.6f'%(np.mean(np.array(synthetic_sample_train)),np.mean(np.array(synthetic_sample_test))))
 




