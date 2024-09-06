import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()    
parser.add_argument('--file_list', type=str, nargs='+', help='file list that contains the lira outputs (for a given model, under a given attack), for plotting the roc curve')
parser.add_argument('--legend_list', type=str, nargs='+', help='legend for each of the file given above')
parser.add_argument('--threshold', type=float, default=1e-3, help='fpr threshold') 
parser.add_argument('--save_tag', type=str, default='tmp', help='file name of the roc figure')  
parser.add_argument('--isPlot', type=int, default=0, help='select 1 to plot the roc figure, otherwise just check the tpr@low fpr') 
parser.add_argument('--title', type=str, default=None, help='figure title')
args = parser.parse_args()

 

fpr_threshold = args.threshold 
file_list = args.file_list
if(args.isPlot): 
    fig, ax = plt.subplots() 

tpr_list = []
fpr_list = []
for i, each in enumerate(file_list):
    #print( each )
    data = np.load(each) 
    y_true = data[ :int(len(data)/2) ]
    y_score = data[ int(len(data)/2): ]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    accuracy = np.max(1-(fpr+(1-tpr))/2)
    auc_score = auc(fpr, tpr) 
    low = tpr[np.where(fpr<fpr_threshold)[0][-1]]

    if(args.isPlot):
        plt.plot(fpr , tpr ) 
    print('%s TPR %.4f @%.4fFPR | AUC %.4f'%(each, low*100, fpr_threshold*100, auc_score)) 


import sys
if(not args.isPlot):
    sys.exit()

legends=args.legend_list
plt.semilogx()
plt.xlim(1e-4,1)        
plt.ylim(0,1.03)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate") 

plt.subplots_adjust(bottom=.25, left=.25, top=.75, right=.75)
legendsize=13
loc = 'best'
plt.legend(labels =legends, loc=loc,ncol=1,fancybox=True, shadow=False, prop={'size': legendsize})

if(args.title!=None):
    plt.title(args.title, fontsize=15, fontweight="normal")
plt.savefig(os.path.join('./', '%s.pdf'%args.save_tag), bbox_inches="tight", dpi=300)
 






