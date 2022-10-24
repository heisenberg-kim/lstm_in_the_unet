
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shutil import copyfile

path_dir_noise = 'path to noise files'
path_dir_clean = 'path to clean files'
path_dir_noisy = 'path to noisy files'

path_to_selected_noise = 'path to uniformly selected by SNR distribution noise'
path_to_selected_clean = 'path to uniformly selected by SNR distribution clean'
path_to_selected_noisy = 'path to uniformly selected by SNR distribution noisy'

file_list_noise = os.listdir(path_dir_noise)
file_list_clean = os.listdir(path_dir_clean)
file_list_noisy = os.listdir(path_dir_noisy)



snr = []
entry = []
num_entry = 0

# extract SNR from name of synthesized noisy files
for item in file_list_noisy:
    snr.append(int(item[item.find('r')+1:item.find('.')]))

# count how many entries exist for each SNR values
for i in range(min(snr), max(snr)+1):
    entry.append(snr.count(i))

#make a directories for selected files
if not os.path.exists(path_to_selected_noise):
    os.makedirs(path_to_selected_noise)
if not os.path.exists(path_to_selected_clean):
    os.makedirs(path_to_selected_clean)
if not os.path.exists(path_to_selected_noisy):
    os.makedirs(path_to_selected_noisy)

# gather files until # of files for each SNR get minimum value of list of entry(baseline = min(entry))
for i in range(min(snr), max(snr)+1):
    for j in range(len(file_list_noisy)):
        if snr[j] == i and num_entry < min(entry):
            print(os.path.join(path_dir_noise, file_list_noise[j]))
            copyfile(os.path.join(path_dir_noise, file_list_noise[j]), os.path.join(path_to_selected_noise, file_list_noise[j]))
            copyfile(os.path.join(path_dir_clean, file_list_clean[j]), os.path.join(path_to_selected_clean, file_list_clean[j]))
            copyfile(os.path.join(path_dir_noisy, file_list_noisy[j]), os.path.join(path_to_selected_noisy, file_list_noisy[j]))
            num_entry += 1
        elif snr[j] == i and num_entry == min(entry):
            print(snr[j])
            break
    num_entry = 0

#draw a histogram for uniformly distributed dataset
selected_snr=[]
file_list_selected_noisy = os.listdir(path_to_selected_noisy)
for item in file_list_selected_noisy:
    selected_snr.append(int(item[item.find('r')+1:item.find('.')]))

plt.hist(selected_snr, bins=np.arange(-5, 26, 1))
#plt.yscale('log')
plt.show()
