import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.core.fromnumeric import shape
import seaborn as sns
import pandas as pd 
sns.set()
facecolor = (245/256,245/256,250/256)
sns.set(rc={'axes.facecolor':facecolor})
plt.style.use("matplotlib_style.mplstyle")
from scipy.constants import Boltzmann as k
from scipy.constants import elementary_charge as q
from numpy.random import poisson
from tqdm import tqdm

gridcolor =  (205/256,205/256,205/256)
mpl.rcParams.update({"axes.grid" : True, "grid.color": gridcolor})

SNSPD_color = 'g'
electron_charge = 1.602e-19
photon_energy = 1.28e-19
simulation_linestyle = 'dashed'

######## SNSPDs ############
sim = np.load("pure_shot_noise_no_integration.npy",allow_pickle=True).item()
nph_sim = sim['num_photon_sweep']
acc_sim = sim['accuracy']

### Plot the experimental results on top of these
data = pd.read_csv("optical_mnist_SNSPD.csv").to_numpy()
voltage = data[:,0]
optical_acc = data[:,2]
voltage_per_photon  = 0.0395
nph_exp = voltage/voltage_per_photon

# plt.semilogx(nph_exp*photon_energy,optical_acc,"*",c=SNSPD_color)
# plt.semilogx(nph_sim*photon_energy,acc_sim,c=SNSPD_color,linestyle=simulation_linestyle,label='_nolegend_')

plt.figure(figsize=(10,8),dpi=80)
ax = plt.gca()
plt.semilogx(nph_sim,acc_sim,linewidth=5)
plt.semilogx(nph_exp,optical_acc,"*",markersize=14)

plt.xlim(1e-2,1e2)

def photons2energy(x):
    photonenergy = 1.28e-19 #Photon energy at 1550nm in Joules
    return x * photonenergy


def energy2photon(x):
    photonenergy = 1.28e-19
    return x /photonenergy

ax2 = ax.secondary_xaxis('top', functions=(photons2energy, energy2photon))
ax2.set_xlabel("Optical Energy per MAC (J)",fontsize=26,labelpad=20)
plt.ylabel("Accuracy",fontsize=26)
plt.xlabel("Number of photons per MAC",fontsize=26)
ax.tick_params(axis='x',labelsize=22)
ax.tick_params(axis='y',labelsize=22)
ax2.tick_params(axis='x',labelsize=22)
ax2.tick_params(axis='y',labelsize=22)
# plt.title("Shot noise limited accuracy, 3 Layer MNIST",fontsize=30)
plt.legend(["Simulation","Experiment"],fontsize=26)
plt.tight_layout()
plt.savefig("shot_noise_limited_MNIST.png")
plt.savefig("shot_noise_limited_MNIST.svg")
plt.savefig("shot_noise_limited_MNIST.pdf")
plt.show()

