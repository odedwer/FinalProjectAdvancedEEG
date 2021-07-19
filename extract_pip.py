# %% imports
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
import pandas as pd
import mne
import mne.decoding as mnd
import matplotlib.pyplot as plt
import scipy
import preprocess_utilities as pu
from importlib import reload
from tqdm import tqdm
import multiprocessing as mp


# %% functions
def load_data_clean(subject_number) -> mne.io.Raw:
    filename = f"PVI_EEGjunior_S{subject_number}_1-clean-raw.fif"
    raw: mne.io.Raw = mne.io.read_raw_fif(filename, preload=True)
    eog_list = ['Nose', 'LHEOG', 'RHEOG', 'RVEOGI']
    emg_list = ['LeftEMGprox', 'rightEMGprox', 'rightEMGdist', 'LeftEMGdist']
    pu.set_ch_types(raw, eog_list, 'eog')
    pu.set_ch_types(raw, emg_list, 'emg')
    raw = raw.set_montage('biosemi64')
    raw.interpolate_bads()
    raw.filter(l_freq=.5, h_freq=50, method='iir')
    return raw


def get_prime_epochs(raw, events):
    event_dict = {"prime_R": 12, "prime_L": 11}
    return mne.Epochs(raw, events, event_dict, tmin=-.4, tmax=1.1, baseline=(-.4, -.1),
                      preload=True)  # , reject={'eeg': 400e-6})


def get_cue_epochs(raw, events):
    event_dict = {"cue_R_reg": 112, "cue_L_reg": 111}
    return mne.Epochs(raw, events, event_dict, tmin=-.4, tmax=1.1, baseline=(-.4, -.1),
                      preload=True)  # , reject={'eeg': 400e-6})


def get_prime_cue_combination_trials(cue_epochs, prime_epochs):
    rr_trials = (cue_epochs.events[:, 2] == 112) & (prime_epochs.events[:, 2] == 12)
    ll_trials = (cue_epochs.events[:, 2] == 111) & (prime_epochs.events[:, 2] == 11)
    rl_trials = (cue_epochs.events[:, 2] == 111) & (prime_epochs.events[:, 2] == 12)
    lr_trials = (cue_epochs.events[:, 2] == 112) & (prime_epochs.events[:, 2] == 11)
    return rr_trials, ll_trials, rl_trials, lr_trials


def get_trial_pip_score(args):
    epochs_data, labels, epoch_vec, trial_num = args
    lda = LinearDiscriminantAnalysis()
    epoch_vec[trial_num] = False
    train_x, train_y = epochs_data[epoch_vec, ...], labels[epoch_vec]
    test_x, test_y = epochs_data[~epoch_vec, ...], labels[~epoch_vec]
    probs = np.zeros((epochs_data.shape[2], 2))
    # create classifier for each timepoint
    for timepoint in range(epochs_data.shape[2]):
        lda.fit(train_x[:, :, timepoint], train_y)
        probs[timepoint, :] = lda.predict_proba(test_x[:, :, timepoint])
    # reset the boolean vector
    probs_ratio = probs[:, test_y] / probs[:, 1 - test_y]
    return np.mean(probs_ratio), np.median(probs_ratio)


def prepare_data_for_pip_scoring(epochs):
    epochs_data = epochs.get_data('eeg')
    epochs_vec = np.ones(epochs_data.shape[0], dtype=bool)
    zero_idx = np.where(epochs.times == 0)[0].item()
    max_time = 0.6  # s
    max_idx = np.round(raw.info['sfreq'] * max_time).astype(int)
    epochs_data = epochs_data[:, :, zero_idx:zero_idx + max_idx]
    for i in range(epochs_data.shape[2]):
        epochs_data[:, :, i] = StandardScaler().fit_transform(epochs_data[:, :, i])
    return epochs_data, epochs_vec


def get_cue_epochs_and_labels(prime_epochs, cong_trials, incong_trials):
    labels = prime_epochs.events[:, 2].copy()
    labels[incong_trials] = 0  # set rl trials to class 0
    labels[cong_trials] = 1  # set ll trials to class 1
    prime_epochs_c = prime_epochs[(labels == 0) | (labels == 1)]
    labels_c = labels[(labels == 0) | (labels == 1)]
    return prime_epochs_c, labels_c


# %% prepare data for pip extraction
if __name__ == "__main__":
    s = 332
    raw = load_data_clean(s)
    event_dict_reg = {"prime_R_reg": 12, "prime_L_reg": 11}
    events = mne.find_events(raw, mask=255)

    # do incongruent VS congruent decoding on right and left cues
    prime_epochs = get_prime_epochs(raw, events)
    cue_epochs = get_cue_epochs(raw, events)

    rr_trials, ll_trials, rl_trials, lr_trials = get_prime_cue_combination_trials(cue_epochs, prime_epochs)
    # extract epochs for trials where the cue was left/right
    prime_epochs_lc, labels_lc = get_cue_epochs_and_labels(prime_epochs, ll_trials, rl_trials)
    prime_epochs_rc, labels_rc = get_cue_epochs_and_labels(prime_epochs, rr_trials, lr_trials)

    epochs_data_lc, epoch_vec_lc = prepare_data_for_pip_scoring(prime_epochs_lc)
    epochs_data_rc, epoch_vec_rc = prepare_data_for_pip_scoring(prime_epochs_rc)
    pool = mp.Pool()
    lc_pool_data = [(epochs_data_lc, labels_lc, epoch_vec_lc, i) for i in range(epochs_data_lc.shape[0])]
    rc_pool_data = [(epochs_data_rc, labels_rc, epoch_vec_rc, i) for i in range(epochs_data_rc.shape[0])]
    print("Calculating left condition scores...")
    lc_pip_scores = []
    for res in tqdm(pool.imap(get_trial_pip_score, lc_pool_data), total=epochs_data_lc.shape[0]):
        lc_pip_scores.append(res)
    lc_pip_scores = np.array(lc_pip_scores)
    print("Calculating right condition scores...")
    rc_pip_scores = []
    for res in tqdm(pool.imap(get_trial_pip_score, rc_pool_data), total=epochs_data_rc.shape[0]):
        rc_pip_scores.append(res)
    rc_pip_scores = np.array(rc_pip_scores)

    # %% read xlsx
    WANTED_COLUMNS = ['Subject', 'trial', 'PrimeCanvas', 'TargetCanvas', 'ActualResp', 'ResponseTime']
    COLUMN_DICT = {'Subject': 'subject', 'PrimeCanvas': 'prime', 'TargetCanvas': 'cue', 'ResponseTime': "rt",
                   "ActualResp": "resp"}
    CANVAS_DICT = {1: 'l', 2: 'r', 3: 'fc', 4: 'n'}
    RESPONSE_DICT = {"z": "l", r"{/}": "r", np.nan: "none"}

    b_data = pd.read_excel("PVI_EEGjunior-332.xlsx")
    b_data['trial'] = b_data.index + 1
    b_data = b_data[WANTED_COLUMNS]
    b_data.rename(columns=COLUMN_DICT, inplace=True)
    b_data['resp'].replace(RESPONSE_DICT, inplace=True)
    b_data[['prime', 'cue']] = b_data[['prime', 'cue']].replace(CANVAS_DICT)
    b_data = b_data[b_data['subject'].notna()]
    b_data['subject'] = b_data['subject'].astype('int')
    b_data: pd.DataFrame = b_data.loc[(b_data["prime"] != "n") & (b_data["cue"] != "fc"),]
    b_data.dropna(inplace=True)
    b_data.reset_index(drop=True, inplace=True)
