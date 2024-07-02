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
# pylint: skip-file
# pyformat: disable
import os
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import functools
# Look at me being proactive!
import matplotlib
import argparse
parser = argparse.ArgumentParser()  
parser.add_argument('--shadow_data_path', type=str, default=None, help='path to the scores from the shadow models')
parser.add_argument('--test_data_path', type=str, default=None, help='path to the scores from the target model')
parser.add_argument('--fpr', nargs='*', type=float, default=[1e-3], help='control the TPR under a low FPR threshold')
parser.add_argument('--eval_target_sample', type=int, default=1, help='evaluate the target sample or its corresponding membership-encoding sample') 
parser.add_argument('--plot', type=int, default=0, help='save the roc curve') 
parser.add_argument('--save_tag', type=str, default=None, help='file tag for saving the predicted MI scores and the membership labels')
parser.add_argument('--output_folder', type=str, default='./atk-result', help='directory for saving the roc curve') 
args = parser.parse_args()


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc

def load_data():
    """
    Load our saved scores and then put them into a big matrix.
    """
    global shadow_scores, shadow_labels, poison_pos, test_scores, test_labels
    print('==> ', args.test_data_path)
    shadow_scores = []  # 
    shadow_labels = []  # 

    test_scores = []    # contain both member and non-member on the target model 
    test_labels = []

    if(args.eval_target_sample):
        file_tag = 'shadow_sample_score'
    else:
        file_tag = 'synthetic_sample_score'

    if(args.shadow_data_path!=None):
        for r, d, f in os.walk(args.shadow_data_path):
            for file in f: 
                if(file_tag in file): 
                    loaded_scores = np.load(os.path.join(r, file))
                    shadow_scores.append( loaded_scores )
                    shadow_labels.append( np.load(os.path.join(r, file.replace(file_tag, 'keep'))) )

        shadow_scores = np.array(shadow_scores)
        shadow_labels = np.array(shadow_labels)
        shadow_labels = shadow_labels.astype(bool)

    for r, d, f in os.walk(args.test_data_path):
        for file in f: 
            if(file_tag in file):
                loaded_scores = np.load(os.path.join(r, file))
                test_scores.append(loaded_scores)
                test_labels.append( np.load(os.path.join(r, file.replace(file_tag, 'keep'))) )
 
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)
    test_labels = test_labels.astype(bool)



def generate_ours(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []
    for j in range(scores.shape[1]): 
        dat_in.append(scores[keep[:,j],j,:])
        dat_out.append(scores[~keep[:,j],j,:])

    in_size = min(min(map(len,dat_in)), in_size)
    out_size = min(min(map(len,dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores): 
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in+1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)
        score = pr_in-pr_out 
        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def generate_ours_offline( keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,fix_variance=False):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []
    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :]) # take all the in_model for that particular sample (keep is used as a bool to identify in_model)
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len,dat_out)), out_size)
    dat_out = np.array([x[:out_size] for x in dat_out])
    mean_out = np.median(dat_out, 1)
    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30) 
        prediction.extend(score.mean(1)) 
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores): 
        prediction.extend(-sc.mean(1))
        answers.extend(ans)
    return prediction, answers

def do_plot(fn, keep, scores, test_keep=None, test_scores=None, legend='', metric='auc', sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    all_predictions, all_answers = fn(keep,
                                   scores, # used to derive the IN/OUT dist
                                   test_keep,  # for testing
                                   test_scores)
    all_predictions = np.array(all_predictions)
    all_answers = np.array(all_answers, dtype=bool)

    if(args.save_tag!=None):
        if(not os.path.exists(args.output_folder)):
            os.mkdir(args.output_folder)
        np.save( os.path.join( args.output_folder, 'lira-%s-%s.npy'%(legend,args.save_tag) ), np.r_[ all_answers, all_predictions*-1] )


    fpr, tpr, auc, acc = sweep_fn(all_predictions, all_answers)
    fpr_thres = args.fpr
    for fp in fpr_thres:
        low = tpr[np.where(fpr<fp)[0][-1]]
        print('\tAttack %s   AUC %.4f, Accuracy %.4f, TPR@%.4f%%FPR is %.4f'%(legend, auc,acc, fp*100, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc
    plt.plot(fpr, tpr, label=legend+metric_text, **plot_kwargs)
    return (acc,auc)


def fig_fpr_tpr():

    plt.figure(figsize=(4,3))
 
    global shadow_scores, shadow_labels, test_scores, test_labels

    do_plot(generate_global,
            keep= shadow_labels,
            scores= shadow_scores,
            test_keep = test_labels,
            test_scores = test_scores,
            legend="global-threshold",
            metric='auc'
    ) 

    if(args.shadow_data_path!=None):
        # this requires shadow models for calibration
        do_plot(functools.partial(generate_ours, fix_variance=False),
                keep= shadow_labels,
                scores= shadow_scores,
                test_keep = test_labels,
                test_scores = test_scores,
                legend="dynamic-variance",
                metric='auc'
        )
         
        do_plot(functools.partial(generate_ours, fix_variance=True),
                keep= shadow_labels,
                scores= shadow_scores,
                test_keep = test_labels,
                test_scores = test_scores,
                legend="fixed-variance",
                metric='auc'
        )

    if(args.plot):
        plt.semilogx()
        plt.xlim(1e-3,1)
        plt.ylim(0,1)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot([0, 1], [0, 1], ls='--', color='gray')
        plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
        plt.legend(fontsize=8)
        plt.savefig("./%s-fprtpr.png"%args.shadow_data_path.replace('/', ''))
        

import sys
if __name__ == '__main__': 
    load_data()
    fig_fpr_tpr()
