import numpy as np
import pandas as pd
import mne
import mne.decoding as mnd
import matplotlib.pyplot as plt
import scipy
import preprocess_utilities as pu
from importlib import reload
from tqdm import tqdm


# %% functions
def load_data(subject_number) -> mne.io.Raw:
    filename = f"PVI_EEGjunior_S{subject_number}_1.bdf"
    raw: mne.io.Raw = mne.io.read_raw_bdf(filename, preload=True)
    eog_list = ['Nose', 'LHEOG', 'RHEOG', 'RVEOGI']
    emg_list = ['LeftEMGprox', 'rightEMGprox', 'rightEMGdist', 'LeftEMGdist']
    pu.set_ch_types(raw, eog_list, 'eog')
    pu.set_ch_types(raw, emg_list, 'emg')
    raw = raw.set_montage('biosemi64')
    return raw


# %% load data
raw = load_data(331)
# raw.plot_psd()
# raw_for_ica = raw.copy()
# raw_for_ica.notch_filter(57.5, method='iir')
# raw_for_ica.notch_filter(60, method='iir')
# raw_for_ica.notch_filter(75, method='iir')
# raw_for_ica.filter(l_freq=1, h_freq=100, method='iir')
events = mne.find_events(raw, mask=255, mask_type='and')

# raw = pu.annotate_bads_auto(raw, 300e-6, 3000e-6)
# %% ica
# ica = mne.preprocessing.ICA(n_components=.95, random_state=97, method='infomax', max_iter=800,
#                             fit_params={'extended': True})
# ica.fit(raw_for_ica, picks=['eeg', 'eog'], reject={'eeg': 300e-6})
# # %% bulk
# ica.plot_components()
# ica.plot_sources(raw_for_ica)
# ica.plot_properties(raw_for_ica, 2)
# ica.apply(raw)
# %%
raw.filter(l_freq=.5, h_freq=50, method='iir')

# %% epoch
# 1 - LEFT
# 2 - RIGHT
# 3 - fC
# 4 - nEUTRAL

event_dict = {"prime_R_reg": 12, "prime_R_fc": 22,
              "prime_L_reg": 11,
              "prime_L_fc": 21}  # , :34, :44, :111, :112, :123, :133, :141, :142, :210, :211, :212, :220, :221, :222
# , :230, :231, :232, :241, :242, :254}
epochs_prime = mne.Epochs(raw, events, event_dict, tmin=-.2, tmax=1.1, baseline=(-.2, -.05), reject={'eeg': 300e-6})
# %% decoding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

# %% TODO - train only LL/RR and test on LR/RL etc.
lda = LinearDiscriminantAnalysis()
clf = make_pipeline(StandardScaler(), lda)
time_decoding = mne.decoding.SlidingEstimator(clf, scoring='roc_auc', verbose=True, n_jobs=8)
labels = epochs_prime.events[:, 2]
labels[(labels == 11) | (labels == 21)] = 1
labels[(labels == 12) | (labels == 22)] = 2
# scores = mne.decoding.cross_val_multiscore(time_decoding, epochs_prime.get_data('eeg'), labels)
# avg_scores = scores.mean(0)
# plt.figure()
# plt.plot(epochs_prime.times, avg_scores)
# plt.hlines(0.5, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)
# %% train on LL and decode LR
# TODO: split to test and train based on LL/RR (train) and LR/RL (test)
event_dict = {"prime_R": 12, "prime_L": 11}
# , :230, :231, :232, :241, :242, :254}
epochs_prime = mne.Epochs(raw, events, event_dict, tmin=-.2, tmax=.13, baseline=(-.2, -.05), reject={'eeg': 300e-6})
event_dict = {"cue_R_reg": 112, "cue_L_reg": 111}
epochs_cue = mne.Epochs(raw, events, event_dict, tmin=-.2, tmax=.13, baseline=(-.2, -.05), reject={'eeg': 300e-6})
rr_trials = (epochs_cue.events[:, 2] == 112) & (epochs_prime.events[:, 2] == 12)
ll_trials = (epochs_cue.events[:, 2] == 111) & (epochs_prime.events[:, 2] == 11)
rl_trials = (epochs_cue.events[:, 2] == 111) & (epochs_prime.events[:, 2] == 12)
lr_trials = (epochs_cue.events[:, 2] == 112) & (epochs_prime.events[:, 2] == 11)
labels = epochs_prime.events[:, 2]
labels[ll_trials | lr_trials] = 1
labels[rr_trials | rl_trials] = 2
train_y = labels[ll_trials | rr_trials]
test_y = labels[lr_trials | rl_trials]
train_x = epochs_prime.get_data('eeg')[rr_trials | ll_trials, :, :]
test_x = epochs_prime.get_data('eeg')[rl_trials | lr_trials, :, :]

# %%
# lda = LinearDiscriminantAnalysis()
# clf = make_pipeline(StandardScaler(), lda)
# time_decoding = mne.decoding.SlidingEstimator(clf, scoring='accuracy', verbose=True, n_jobs=8)
# scores = mne.decoding.cross_val_multiscore(time_decoding, test_x, test_y)
# scores=scores.mean(axis=0)
# plt.figure()
# plt.plot(epochs_prime.times, scores)
# plt.hlines(0.5, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)

# %%
train_x = train_x[:, :, 225:]
test_x = test_x[:, :, 225:]
test_x = test_x.reshape((test_x.shape[0],test_x.shape[1]*test_x.shape[2]))
train_x = train_x.reshape((train_x.shape[0],train_x.shape[1]*train_x.shape[2]))

lda = LinearDiscriminantAnalysis()
permutation_scores = np.zeros(2000)
for i in tqdm(range(permutation_scores.size)):
    lda.fit(train_x, np.random.choice(train_y,train_y.size,False))
    permutation_scores[i] = lda.score(test_x, test_y)
lda.fit(train_x, train_y)
scores = lda.score(test_x, test_y)
plt.hist(permutation_scores,bins=25)
plt.axvline(scores,color="k",linestyle=":",linewidth=3)
print(f'accuracy is {scores}')
print(f'subject p-value from permutation is: {np.mean(scores<permutation_scores)}')
# %%
plt.figure()
plt.plot(epochs_prime.times, scores)
plt.hlines(0.5, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)
