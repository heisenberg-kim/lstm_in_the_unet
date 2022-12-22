'''
this script separates clean(speech) dataset into several clean datasets with different clean level.
in this script, SNR means clean/(clean-evaluated clean).
when u get high SNR value, this clean file is highly clean.
'''

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import shutil


version = "model_ver to save histogram"
path_dir_selected = 'path to directories of selected clean'
path_dir_clean = 'path to clean dataset to select'
path_dir_eval = 'path to evaluation files of clean dataset'
file_list_clean = os.listdir(path_dir_clean)
file_list_eval = os.listdir(path_dir_eval)
snr = []

if os.path.isdir(path_dir_selected + "/clean_selected_1") == False:
    os.mkdir(path_dir_selected + "/clean_selected_1")
if os.path.isdir(path_dir_selected + "/clean_selected_2") == False:
    os.mkdir(path_dir_selected + "/clean_selected_2")
if os.path.isdir(path_dir_selected + "/clean_selected_3") == False:
    os.mkdir(path_dir_selected + "/clean_selected_3")
if os.path.isdir(path_dir_selected + "/clean_selected_4") == False:
    os.mkdir(path_dir_selected + "/clean_selected_4")
#if os.path.isdir(path_dir_selected + "/clean_selected_5") == False:
#    os.mkdir(path_dir_selected + "/clean_selected_5")
if os.path.isdir(path_dir_selected + "/clean_selected_45") == False:
    os.mkdir(path_dir_selected + "/clean_selected_45")
#if os.path.isdir(path_dir_selected + "/clean_selected_50") == False:
#    os.mkdir(path_dir_selected + "/clean_selected_50")

count=0
for i in range(len(file_list_clean)):
    count += 1    
    print("{}/{}".format(count, len(file_list_clean)))
    file_path_clean = path_dir_clean + "/" + file_list_clean[i]
    file_path_eval = path_dir_eval + "/" + file_list_eval[i]
    file_path_selected_1 = path_dir_selected + "/clean_selected_1/" + file_list_clean[i]
    file_path_selected_2 = path_dir_selected + "/clean_selected_2/" + file_list_clean[i]
    file_path_selected_3 = path_dir_selected + "/clean_selected_3/" + file_list_clean[i]
    file_path_selected_4 = path_dir_selected + "/clean_selected_4/" + file_list_clean[i]
    #file_path_selected_45 = path_dir_selected + "/clean_selected_5/" + file_list_clean[i]
    file_path_selected_45 = path_dir_selected + "/clean_selected_45/" + file_list_clean[i]
    #file_path_selected_50 = path_dir_selected + "/clean_selected_50/" + file_list_clean[i]
    clean, sr = librosa.load(file_path_clean, sr=16000)
    eval, sr = librosa.load(file_path_eval, sr=16000)

    diff = np.array(clean[:-128] - eval)
    #diff = np.array(clean[128:] - eval)
    #diff_abs = np.abs(diff)
    diff_sq = diff ** 2
    diff_mean = np.mean(diff_sq)

    true = np.array(clean)
    #true_abs = np.abs(true)
    true_sq = true ** 2
    true_mean = np.mean(true_sq)

    snr_value = np.log10(true_mean / diff_mean)

#	if snr_value >= 5:
#		shutil.copyfile(file_path_clean, file_path_selected_50)
#		print(snr_value)
#	elif snr_value >=4.5 and snr_value <5:
#		shutil.copyfile(file_path_clean, file_path_selected_45)
#		print(snr_value)
#	else:
#			continue

    if snr_value >= 4:
    	shutil.copyfile(file_path_clean, file_path_selected_45)
    elif snr_value >= 3:
        shutil.copyfile(file_path_clean, file_path_selected_4)
    elif snr_value >= 2:
        shutil.copyfile(file_path_clean, file_path_selected_3)
    elif snr_value >= 1:
        shutil.copyfile(file_path_clean, file_path_selected_2)
    else:
        shutil.copyfile(file_path_clean, file_path_selected_1)
    snr.append(snr_value)
	
print("min:{}".format(min(snr)))
print("max:{}".format(max(snr)))

#plt.hist(snr)
plt.hist(snr, bins=30, log=True)

plt.xlabel("SNR: clean/(clean-estimated_clean)")
plt.ylabel("entries")
plt.title("SNR distribution of clean : DRY Speech?")

#plt.show()
plt.savefig(version + '.png')

