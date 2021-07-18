from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
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
def load_data_clean(subject_number) -> mne.io.Raw:
    filename = f"PVI_EEGjunior_S{subject_number}_1-clean-raw.fif"
    raw: mne.io.Raw = mne.io.read_raw_fif(filename, preload=True)
    eog_list = ['Nose', 'LHEOG', 'RHEOG', 'RVEOGI']
    emg_list = ['LeftEMGprox', 'rightEMGprox', 'rightEMGdist', 'LeftEMGdist']
    pu.set_ch_types(raw, eog_list, 'eog')
    pu.set_ch_types(raw, emg_list, 'emg')
    raw = raw.set_montage('biosemi64')
    return raw


# %% load data
average_decoding_congruency = []
subject_TOI = []  # add timepoints from which we will extract the PIP
subjects = [331, 332, 333, 334, 335, 336, 337, 338, 339, 342, 343, 344, 345, 346, 347, 349, 349]
for s in subjects:
    raw = load_data_clean(s)
    raw.interpolate_bads()

    raw.filter(l_freq=.5, h_freq=50, method='iir')

    # epoch
    # 1 - LEFT
    # 2 - RIGHT
    # 3 - fC
    # 4 - nEUTRAL

    event_dict = {"prime_R_reg": 12, "prime_R_fc": 22,
                  "prime_L_reg": 11,
                  "prime_L_fc": 21}
    event_dict_reg = {"prime_R_reg": 12, "prime_L_reg": 11}
    events = mne.find_events(raw, mask=255)
    epochs_prime = mne.Epochs(raw, events, event_dict_reg, tmin=-.2, tmax=1.1, baseline=(-.2, -.05),
                              reject={'eeg': 300e-6})
    #  decoding

    #  how long does it take for visual information to appear?
    # lda = LinearDiscriminantAnalysis()
    # clf = make_pipeline(StandardScaler(), lda)
    # time_decoding = mne.decoding.SlidingEstimator(clf, scoring='accuracy', verbose=True, n_jobs=8)
    # scores = mne.decoding.cross_val_multiscore(time_decoding, epochs_prime.get_data('eeg'), epochs_prime.events[:, 2])
    # scores = scores.mean(axis=0)
    # plt.figure()
    # plt.plot(epochs_prime.times, scores)
    # plt.hlines(0.5, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)
    #
    #  TODO - train only LL/RR and test on LR/RL etc.
    # labels = epochs_prime.events[:, 2]
    # labels[(labels == 11) | (labels == 21)] = 1
    # labels[(labels == 12) | (labels == 22)] = 2
    # scores = mne.decoding.cross_val_multiscore(time_decoding, epochs_prime.get_data('eeg'), labels)
    # avg_scores = scores.mean(0)
    # plt.figure()
    # plt.plot(epochs_prime.times, avg_scores)
    # plt.hlines(0.5, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)
    #
    # TODO: split to test and train based on LL/RR (train) and LR/RL (test)
    event_dict = {"prime_R": 12, "prime_L": 11}
    # do incongruent VS congruent decoding on right and left cues
    epochs_prime = mne.Epochs(raw, events, event_dict, tmin=-.4, tmax=1.1, baseline=(-.4, -.1),
                              preload=True)  # , reject={'eeg': 400e-6})
    event_dict = {"cue_R_reg": 112, "cue_L_reg": 111}
    epochs_cue = mne.Epochs(raw, events, event_dict, tmin=-.4, tmax=1.1, baseline=(-.4, -.1),
                            preload=True)  # , reject={'eeg': 400e-6})
    # create classifier
    lda = LinearDiscriminantAnalysis()
    clf = make_pipeline(StandardScaler(), lda)
    time_decoding = mne.decoding.SlidingEstimator(clf, scoring='roc_auc', verbose=True, n_jobs=12)
    kfold = 17
    #  left
    rr_trials = (epochs_cue.events[:, 2] == 112) & (epochs_prime.events[:, 2] == 12)
    ll_trials = (epochs_cue.events[:, 2] == 111) & (epochs_prime.events[:, 2] == 11)
    rl_trials = (epochs_cue.events[:, 2] == 111) & (epochs_prime.events[:, 2] == 12)
    lr_trials = (epochs_cue.events[:, 2] == 112) & (epochs_prime.events[:, 2] == 11)
    labels = epochs_prime.copy().events[:, 2]
    labels[rl_trials] = 0
    labels[ll_trials] = 1
    epochs_prime_lc = epochs_prime[(labels == 0) | (labels == 1)]
    labels_lc = labels[(labels == 0) | (labels == 1)]
    scores_lc = mne.decoding.cross_val_multiscore(estimator=time_decoding, X=epochs_prime_lc.get_data('eeg'),
                                                  y=labels_lc, cv=kfold)
    scores_lc = np.mean(scores_lc,axis=0)
    #  right
    labels = epochs_prime.copy().events[:, 2]
    labels[lr_trials] = 0
    labels[rr_trials] = 1
    epochs_prime_rc = epochs_prime[(labels == 0) | (labels == 1)]
    labels_rc = labels[(labels == 0) | (labels == 1)]
    scores_rc = mne.decoding.cross_val_multiscore(time_decoding, epochs_prime_rc.get_data('eeg'), labels_rc, cv=kfold)
    scores_rc = np.mean(scores_rc,axis=0)

    # plt.figure()
    # plt.plot(epochs_prime.times, scores_lc, label="left cue")
    # plt.hlines(0.5, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)
    # plt.plot(epochs_prime.times, scores_rc, label="right cue")
    # plt.plot(epochs_prime.times, (scores_rc+scores_lc)/2, label="average")
    # plt.legend()
    # plt.title(f'Incongruent VS congruent trials decoding for left cues and right cues, subject {s}')
    average_decoding_congruency.append((scores_rc + scores_lc) / 2)
    move_avg_auc = np.convolve(average_decoding_congruency[-1],np.ones(10)/10) #moving average over 5 points
    # plt.figure()
    # plt.plot(epochs_prime.times, scores_lc, label="left cue")
    # plt.hlines(0.5, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)
    # plt.plot(epochs_prime.times, scores_rc, label="right cue")
    # plt.plot(epochs_prime.times, (scores_rc+scores_lc)/2, label="average")
    # plt.legend()
    # plt.title(f'Incongruent VS congruent trials decoding for left cues and right cues, subject {s}')

mean_subjects_average_decoding_congruency = np.mean(average_decoding_congruency, axis=0)
mean_subjects_average_decoding_congruency = average_decoding_congruency[0]
# plt.figure()
plt.plot(epochs_prime.times, mean_subjects_average_decoding_congruency, color='grey', alpha=0.1)
plt.hlines(0.5, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)
plt.title(f'Incongruent VS congruent trials decoding for left cues and right cues, average of 331-338')

# cross decoding
train_y = labels[ll_trials | rr_trials]
test_y = labels[lr_trials | rl_trials]
train_x = epochs_prime.get_data('eeg')[rr_trials | ll_trials, :, :]
test_x = epochs_prime.get_data('eeg')[rl_trials | lr_trials, :, :]

# %%
train_x = train_x[:, :, 225:]
test_x = test_x[:, :, 225:]
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1] * test_x.shape[2]))
train_x = train_x.reshape((train_x.shape[0], train_x.shape[1] * train_x.shape[2]))

lda = LinearDiscriminantAnalysis()
permutation_scores = np.zeros(2)
for i in tqdm(range(permutation_scores.size)):
    lda.fit(train_x, np.random.choice(train_y, train_y.size, False))
    permutation_scores[i] = lda.score(test_x, test_y)
lda.fit(train_x, train_y)
scores = lda.score(test_x, test_y)
plt.hist(permutation_scores, bins=25)
plt.axvline(scores, color="k", linestyle=":", linewidth=3)
print(f'accuracy is {scores}')
print(f'subject p-value from permutation is: {np.mean(scores < permutation_scores)}')
# %%
plt.figure()
plt.plot(epochs_prime.times, scores)
plt.hlines(0.5, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)
