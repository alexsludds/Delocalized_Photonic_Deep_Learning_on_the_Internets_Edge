from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt 

name = "lincoln"

data = np.load("noise_robust_mnist_weights.npy",allow_pickle=True).item()
x_test = data['xtest']
ytest = data['ytest']
layer1_weights = data["layer1"]
layer2_weights = data['layer2']
layer3_weights = data['layer3']
b1 = data['b1']
b2 = data['b2']
b3 = data['b3']

############## THREE LAYER OPTICAL
#Here we are going to import the three layer data from saved_data_three_layer
out = np.load("98_8_lincoln.npy",allow_pickle=True).item()
num_to_classify = out['optical_output_activations'].shape[0]

def get_acc(out,start=0):
    want_to_save = out['optical_output_activations']
    want_to_save = np.argmax(want_to_save,axis=1)
    num_correct = 0
    
    for index, value in enumerate(want_to_save):
        if value == np.argmax(ytest[index]):
            
            num_correct += 1
    return num_correct

su = 0
su += get_acc(out)
optical_system_accuracy = su/num_to_classify
print("Optical System Accuracy: ",optical_system_accuracy)

#Here we want to create the "misclassification" matrix of this Netcast system
num_in_each_class = np.zeros(10)
misclassification_matrix = np.zeros((10,10)) #first dim actual, second dim measured
for i in range(num_to_classify):
    start = 0
    correct_class =  np.argmax(ytest[i])
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
tick_fontsize = 30
label_y_position = -0.055
matshow_fontsize = 26

default_flier_props = dict(markerfacecolor='moccasin', marker='o')

##### LEFT COLUMN Netcast OUTPUT SCORES ########################
ax = plt.subplot(gsleft[0, 0])
matshow = ax.matshow(misclassification_matrix.T, cmap=plt.cm.YlGn,vmin=0,vmax=1) #Can use plt.cm.Blues

cbar = fig.colorbar(matshow,ax=ax,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=26) 
#NOTE!!!!! Imshow flips the axis on the data

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

ax.set_title("Deployed Fiber: " + "{:.1f}".format(optical_system_accuracy*100) + "% Accurate",fontsize=title_fontsize,pad=40)

plt.tight_layout()
plt.savefig("lincoln.png",dpi=200,bbox_inches='tight')
plt.savefig("lincoln.pdf",dpi=200,bbox_inches='tight')
plt.savefig("lincoln.svg",dpi=200,bbox_inches='tight')
plt.show(block=True)