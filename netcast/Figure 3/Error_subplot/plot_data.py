import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = np.load("diff_dist_2022-02-14_161538.npy",allow_pickle=True).item()

optics = data['optical_storage']
electronics = data['ground_truth_storage']

std_dev_error = np.std(electronics-optics)
#Calc rmse 
temp = 0
for i in range(electronics.size):
    temp += (electronics[i] - optics[i])**2
rmse = np.sqrt(temp / electronics.size)

labelsize = 20
# start with a square Figure
fig = plt.figure(figsize=(12, 8),dpi=80)

# Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(7, 3), height_ratios=(7, 3),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

scatterax = fig.add_subplot(gs[0, 0])
scatterax.scatter(electronics, optics,s=1)
scatterax.tick_params(axis="x",labelbottom=False)
scatterax.tick_params(axis='y',labelsize=20)
scatterax.set_ylabel(r"$\rm{\hat{y}}$",fontsize=labelsize)
# scatterax.xaxis.set_ticklabels([])
# ax_hist = fig.add_subplot(gs[0, 0], sharex=ax)
ax_error = fig.add_subplot(gs[1, 0], sharex=scatterax)
ax_error.plot(electronics,electronics-optics,"o",markersize=1)
ax_error.set_xlabel(r"$\rm{y}$",fontsize=labelsize)
ax_error.set_ylabel(r"$\rm{y} - \rm{\hat{y}}$",fontsize=labelsize)
ax_error.tick_params(axis='y',labelsize=20)
ax_error.tick_params(axis='x',labelsize=20)
ax_error.set_ylim(-0.03,0.03)
# use the previously defined function


ax_hist = fig.add_subplot(gs[1,1],sharey=ax_error)
ax_hist.tick_params(axis="y", labelleft=False)
ax_hist.hist(electronics-optics,orientation='horizontal',bins=25,lw=0.2)
ax_hist.tick_params(axis='x',labelsize=20)
ax_hist.set_xlabel("Frequency",fontsize=labelsize)
ax_hist.annotate(r"", xy=(425,0.0075), xytext=(425, -0.0065), arrowprops=dict(arrowstyle="<->",facecolor='black',edgecolor='black',lw=1.2))
bbox_props = dict(boxstyle="round", fc="w", ec="0.1", alpha=0.6)
ax_hist.text(805, 0.013, r"$\sigma_{\rm{rms}} = $" + str(format(std_dev_error, '.3f')), ha="center", va="center", size=16,bbox=bbox_props)
plt.savefig("error_subplot.png")
plt.savefig("error_subplot.pdf")
plt.savefig("error_subplot.svg")
plt.show()
