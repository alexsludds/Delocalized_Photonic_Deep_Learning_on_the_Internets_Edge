#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm
import pandas as pd 
from scipy import interpolate
from scipy.interpolate import interp1d

#%%

######## CONSTANTS ###########
num_samples = 10000
ivc102_capacitance = 1e-11
energy_per_mac_sweep = np.logspace(-18,-12,num=400)
electron_charge = 1.602e-19
photon_energy = 1.28e-19
voltage_per_code = 0.2/2**(15)

########## COLOR CODING AND STYLING ########
N_1_color = 'teal'
time_integration_color = 'b'
APD_color = 'r'
SNSPD_color = 'g'
koheron_color = (173/255, 157/255, 35/255)
thorlabs_color = 'c'

simulation_linestyle = 'dashed'

labelsize = 24
titlesize = 24
tickssize = 22
legendsize = 16

plt.figure(figsize=(14,8),dpi=120)
ax = plt.gca()

############# Commercial off the shelf photodetectors #######
#### PLOTTING FROM SNR_EXPERIMENT_SLOW_LiNbO3
thorlabs_data = pd.read_csv("thorlabs_SNR_measurement.csv")
thorlabs_data = thorlabs_data.to_numpy()
thorlabs_data = thorlabs_data[np.abs(thorlabs_data[:,2]) < 0.01]

#Back of the envelope for difference distribution accuracy
RMS_noise = 300e-6/2 #Factor of 2 for 50 ohm termination
bandwidth = 775e3
gain = 24e3 #V/A or V/W since 1A/W
inp = np.random.rand(num_samples)
weight = np.random.rand(num_samples)

output = np.dot(inp, weight)
thorlabs_noise_storage = []
averaging = 10
for e in tqdm(energy_per_mac_sweep):
    temp = 0
    for _ in range(averaging):
        optical_power = e*bandwidth
        voltage = gain * optical_power
        SNR = voltage/RMS_noise
        optical_output = output + 1./SNR * (np.random.rand(num_samples) - 0.5)
        difference = output - optical_output
        temp += np.std(difference)
    thorlabs_noise_storage.append(temp/averaging)

# Do the same but for the Koheron PD
#### PLOTTING FROM SNR_EXPERIMENT_SLOW_LiNbO3
koheron_data = pd.read_csv("koheron_SNR_measurement.csv")
koheron_data = koheron_data.to_numpy()
koheron_data = koheron_data[np.abs(koheron_data[:,2]) < 0.05]

# Koheron Detector Noise 
koheron_noise_storage = []
RMS_noise = 286e-6 #Calculated from the datasheet, units of volts RMS. 7pA/sqrt(Hz) * sqrt(110MHz) * 3900V/A
bandwidth = 110e6
gain = 3900
averaging = 10
for e in tqdm(energy_per_mac_sweep):
    temp = 0
    for _ in range(averaging):
        optical_power = e*bandwidth
        voltage = gain * optical_power
        SNR = voltage/RMS_noise
        optical_output = output + 1./SNR * (np.random.rand(num_samples) - 0.5)
        difference = output - optical_output
        temp += np.std(difference)
    koheron_noise_storage.append(temp/averaging)

dicty = np.load("acc_storage.npy",allow_pickle=True).item()
dd_storage = dicty['dd_storage']
acc_storage = dicty['acc_storage']

from scipy.interpolate import interp1d
s = interp1d(dd_storage, acc_storage, kind='linear',bounds_error=False)

e1, = plt.semilogx(thorlabs_data[:,0],s(thorlabs_data[:,3]),"*",c=thorlabs_color)
t1, = plt.semilogx(energy_per_mac_sweep,s(thorlabs_noise_storage),c=thorlabs_color,linestyle='--',label='_nolegend_')
e2, = plt.semilogx(koheron_data[:,0],s(koheron_data[:,3]),"*",c=koheron_color)
t2, = plt.semilogx(energy_per_mac_sweep,s(koheron_noise_storage),c=koheron_color,linestyle='--',label='_nolegend_')

####### Linear Mode APD ##########
#Mean optical Energy ,Mean optical power,baseline_accuracy,optical_accuracy
dicty = np.load("APD_storage.npy",allow_pickle=True).item()

acc_storage = dicty['acc_storage']
dd_storage = dicty['dd_storage']
s_n1 = interp1d(dd_storage, acc_storage, kind='linear',bounds_error=False)

data = pd.read_csv("APD_SNR_measurement.csv")
data = data.to_numpy()

mean_threshold_n_1 = 0.3

data = data[np.abs(data[:,2]) < mean_threshold_n_1]
apd_coupling_effiency = 0.5 #Measured 50% coupling effeciency from fiber mode to APD
mean_optical_energy = data[:,0]/apd_coupling_effiency

###### Simulation of noise
num_samples = 10000
inp = np.random.rand(num_samples)
weight = np.random.rand(num_samples)

output = np.dot(inp, weight)
SNR_storage = []
storage = []
energy_per_mac_sweep = np.logspace(-19,-12,num=3000)
integrated_noise_power = 17e-9
bandwidth = 400e6
noise_energy_per_time = integrated_noise_power/bandwidth
averaging = 10
for e in tqdm(energy_per_mac_sweep):
    temp = 0
    for _ in range(averaging):
        SNR = e/noise_energy_per_time
        SNR_storage.append(SNR)
        optical_output = output + 1./SNR * (np.random.rand(num_samples) - 0.5)
        difference = output - optical_output
        temp += np.std(difference)
    storage.append(temp/averaging)

e3, = plt.semilogx(mean_optical_energy,s_n1(data[:,3]),"*",c=APD_color)
t3, = plt.semilogx(energy_per_mac_sweep,s_n1(storage),c=APD_color,linestyle=simulation_linestyle,label='_nolegend_')

#Data taken with ivc102, IR photodiode
######### N = 100 data ############
data = pd.read_csv("integrator_N_100.csv").to_numpy()
#Prune the data! We pre-select data points which have poor 
dd_std = data[:,3]
data = data[dd_std < 3.8, :]
dd_mean = data[:,4]
data = data[np.abs(dd_mean) < 1.2,:]
average_delta_code_per_readout = data[:,0]
average_delta_code_per_MAC = average_delta_code_per_readout/100
average_delta_voltage_per_MAC = average_delta_code_per_MAC * 0.2/2**15
average_delta_charge_per_MAC = 1e-11 * average_delta_voltage_per_MAC
electron_charge = 1.602e-19
photon_energy = 1.28e-19
optical_energy_per_MAC = average_delta_charge_per_MAC*photon_energy/electron_charge
optical_accuracy = data[:,2]
e4, = plt.semilogx(optical_energy_per_MAC,optical_accuracy,"*",c=time_integration_color)

########### Simulation of N = 100 ###########
data = np.load("N_100_simulation_sweep_results.npy",allow_pickle=True).item()

t4, = plt.semilogx(data["energy_per_mac_sweep"],data["ivc_N_100_detector_accuracy_storage"],c=time_integration_color,linestyle=simulation_linestyle,label='_nolegend_')

############# Plotting buracracy ###########
ax.set_xticks([
    1e-18,
    1e-17,
    1e-16,
    1e-15,
    1e-14,
    1e-13,
    1e-12
])

#Add in a twin-x axis
def energytophotons(x):
    return x/photon_energy

def photonstoenergy(x):
    return x*photon_energy

ax2 = ax.secondary_xaxis('top', functions=(energytophotons, photonstoenergy))
ax2.set_xlabel('Number of Photons per MAC',fontsize=labelsize,labelpad=10)
ax2.tick_params(axis='x',labelsize = tickssize)

ax.tick_params(axis='y',labelsize = tickssize)
ax.tick_params(axis='x',labelsize = tickssize)

ax2.set_xticks([
    1e1,
    1e2,
    1e3,
    1e4,
    1e5,
    1e6,
    1e7],)

import matplotlib
import matplotlib.lines as mlines
legend = plt.legend(
    [(t1,t2,t3,t4),e1,e2,e3,e4],
    [
    "Theory",
    "Thorlabs PDA10CS",
    "Koheron PD100",
    "Linear APDs",
    "Time Integrating Receiver"
    ],
    prop={"size":legendsize},loc = 'lower right',fancybox=True, framealpha=0.7,
    handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)}
)
# legend.legendHandles[0]._legmarker.set_markersize(28)
# legend.legendHandles[1]._legmarker.set_markersize(14)
# legend.legendHandles[2]._legmarker.set_markersize(14)
# legend.legendHandles[3]._legmarker.set_markersize(14)
# legend.legendHandles[4]._legmarker.set_markersize(14)

plt.xlim(1e-18,1e-12)
plt.ylabel("MNIST Accuracy",fontsize=labelsize)
plt.xlabel("Optical Energy per MAC (J)",fontsize=labelsize)
plt.tight_layout()
plt.savefig("main_noise_figure.png",dpi='figure')
plt.savefig("main_noise_figure.pdf",dpi='figure')
plt.savefig("main_noise_figure.svg",dpi='figure')
plt.show()

# %%
