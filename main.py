import numpy as np
import pandas as pd
import mne
import mne.decoding as mnd
import matplotlib.pyplot as plt
import scipy
import preprocess_utilities as pu
from importlib import reload


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
event_dict = {"prime_LL": 11, "prime_LR": 12, "prime_RL": 21,
              "prime_RR": 22}  # , :34, :44, :111, :112, :123, :133, :141, :142, :210, :211, :212, :220, :221, :222
# , :230, :231, :232, :241, :242, :254}
epochs_prime = mne.Epochs(raw, events, event_dict, tmin=-.2, tmax=1.1, baseline=(-.2, -.05), reject={'eeg': 300e-6})
# %% decoding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

if __name__ == "__main__":
    # TODO - train only LL/RR and test on LR/RL etc.
    lda = LinearDiscriminantAnalysis()
    clf = make_pipeline(StandardScaler(), lda)
    time_decoding = mne.decoding.SlidingEstimator(clf, scoring='accuracy', verbose=True, n_jobs=8)
    labels = epochs_prime.events[:, 2]
    labels[(labels == 11) | (labels == 12)] = 1
    labels[labels != 1] = 2
    scores = mne.decoding.cross_val_multiscore(time_decoding, epochs_prime.get_data('eeg'), labels)
    avg_scores = scores.mean(0)
    plt.figure()
    plt.plot(epochs_prime.times, avg_scores)
    plt.hlines(0.25, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)
