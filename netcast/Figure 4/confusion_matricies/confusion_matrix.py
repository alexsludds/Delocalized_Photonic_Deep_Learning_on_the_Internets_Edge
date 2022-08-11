from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt 

# name = "high_acc"
# name = "mid_acc"
name = "low_acc"


data = np.load("noise_robust_mnist_weights.npy",allow_pickle=True).item()
x_test = data['xtest']
y_test = data['ytest']
layer1_weights = data["layer1"]
layer2_weights = data['layer2']
layer3_weights = data['layer3']
b1 = data['b1']
b2 = data['b2']
b3 = data['b3']
y_test = np.squeeze(y_test)
############## THREE LAYER OPTICAL
#Here we are going to import the three layer data from saved_data_three_layer
out = np.load(name + ".npy",allow_pickle=True).item()
num_to_classify = out['optical_output_activations'].shape[0]

def get_acc(out,start=0):
    want_to_save = out['optical_output_activations']
    want_to_save = np.argmax(want_to_save,axis=1)
    num_correct = 0
    
    for index, value in enumerate(want_to_save):
        if value == np.argmax(y_test[index]):
            
            num_correct += 1
    return num_correct

su = 0
su += get_acc(out)
print("Optical System Accuracy",su/num_to_classify)

#Here we want to create the "misclassification" matrix of this Netcast system
num_in_each_class = np.zeros(10)
misclassification_matrix = np.zeros((10,10)) #first dim actual, second dim measured
for i in range(num_to_classify):
    start = 0
    correct_class =  np.argmax(y_test[i])
    num_in_each_class[correct_class] += 1
    want_to_save = out['optical_output_activations']
    want_to_save = np.argmax(want_to_save,axis=1)
    guess_class = want_to_save[i]
    misclassification_matrix[correct_class,guess_class] += 1

misclassification_matrix = misclassification_matrix.T/num_in_each_class

misclassification_matrix = misclassification_matrix.T

import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
# gs = gridspec.GridSpec(3, 2,width_ratios=[2,1])
##### PLOTTTING!
# fig, (ax,ax2,ax3,ax4) = plt.subplots(figsize=(21,7),nrows=2,ncols=2)
fig = plt.figure(figsize=(21,14))
# plt.suptitle("MNIST 2 Layer Output Scores",fontsize=26)

gs0 = fig.add_gridspec(nrows=1, ncols=1)

gsleft = gs0[0].subgridspec(1, 1) #row,column
# gsright = gs0[1].subgridspec(3, 1)

color_vmin = 0.0
color_vmax = 1.0
label_fontsize = 32
title_fontsize = 42
tick_fontsize = 26
label_y_position = -0.055
matshow_fontsize = 26

default_flier_props = dict(markerfacecolor='moccasin', marker='o')

##### LEFT COLUMN Netcast OUTPUT SCORES ########################
ax = plt.subplot(gsleft[0, 0])
matshow = ax.matshow(misclassification_matrix.T, cmap=plt.cm.YlGn,vmin=0,vmax=1) #Can use plt.cm.Blues

if name == "high_acc":
    cbar = fig.colorbar(matshow,ax=ax,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=26) 
#NOTE!!!!! Imshow flips the axis on the data

if name !=  'low_acc':
    for i in range(10):
        for j in range(10):
            c = misclassification_matrix[j,i]
            if i != j:
                ax.text(j, i, str("{0:.2f}".format(round(c,2))), va='center', ha='center',fontsize=matshow_fontsize)
            else:
                ax.text(j, i, str("{0:.2f}".format(round(c,2))), va='center', ha='center',color='white',fontsize=matshow_fontsize)

if name == "low_acc":
    #Will decide based on a threshold value what the color should be
    threshold = 0.4
    for i in range(10):
        for j in range(10):
            c = misclassification_matrix[j,i]
            if c < threshold:
                ax.text(j, i, str("{0:.2f}".format(round(c,2))), va='center', ha='center',fontsize=matshow_fontsize)
            else:
                ax.text(j, i, str("{0:.2f}".format(round(c,2))), va='center', ha='center',color='white',fontsize=matshow_fontsize)        


ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.tick_params(axis='x',labelsize=tick_fontsize)
ax.tick_params(axis='y',labelsize=tick_fontsize)
ax.set_ylim(9.5,-0.5)
ax.set_xlim(-0.5,9.5)
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_coords(0.5, label_y_position)
ax.set_xlabel("Actual MNIST number",fontsize=label_fontsize)
ax.set_ylabel("Predicted MNIST number",fontsize=label_fontsize)
if name == "high_acc":
    ax.set_title("99% Classification Accuracy",fontsize=title_fontsize,pad=40)

if name == "mid_acc":
    ax.set_title("79% Classification Accuracy",fontsize=title_fontsize,pad=40)

if name == "low_acc":
    ax.set_title("37% Classification Accuracy",fontsize=title_fontsize,pad=40)

plt.tight_layout()
plt.savefig(name + ".png",dpi=200,bbox_inches='tight')
plt.savefig(name + ".pdf",dpi=200,bbox_inches='tight')
plt.savefig(name + ".svg",dpi=200,bbox_inches='tight')
plt.show(block=False)