# %% imports
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import mne
import matplotlib.pyplot as plt
import preprocess_utilities as pu
from tqdm import tqdm
import seaborn as sns
from typing import Tuple, List


def load_data_clean(subject_number) -> mne.io.Raw:
    """
    Loads the clean data given the subject number. Assumes the data is in the same folder as the code
    :param subject_number: The number of the subject to load
    :return: Raw file of cleaned subject data
    """
    filename = f"PVI_EEGjunior_S{subject_number}_1-clean-raw.fif"
    raw: mne.io.Raw = mne.io.read_raw_fif(filename, preload=True)
    eog_list = ['Nose', 'LHEOG', 'RHEOG', 'RVEOGI']
    emg_list = ['LeftEMGprox', 'rightEMGprox', 'rightEMGdist', 'LeftEMGdist']
    pu.set_ch_types(raw, eog_list, 'eog')
    pu.set_ch_types(raw, emg_list, 'emg')
    raw = raw.set_montage('biosemi64')
    raw.set_eeg_reference()
    raw.interpolate_bads()
    raw.filter(l_freq=.5, h_freq=50, method='iir')
    return raw


def get_decoding_scores(epochs, labels, kfold=17, classifier=LinearDiscriminantAnalysis, scoring='roc_auc',
                        verbose=True, n_jobs=12):
    classier_inst = classifier()
    clf = make_pipeline(StandardScaler(), classier_inst)
    time_decoding = mne.decoding.SlidingEstimator(clf, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
    scores = mne.decoding.cross_val_multiscore(estimator=time_decoding, X=epochs.get_data('eeg'), y=labels,
                                               cv=kfold).mean(0)
    return scores


def get_prime_epochs(raw, events, tmin=-.4, tmax=1.1, baseline=(-.4, -.1)) -> mne.Epochs:
    """
    Extract epochs around the prime
    :return:
    """
    event_dict = {"prime_R": 12, "prime_L": 11, "prime_R_N": 22, "prime_L_N": 21}
    return mne.Epochs(raw, events, event_dict, tmin=tmin, tmax=tmax, baseline=baseline,
                      preload=True)  # , reject={'eeg': 400e-6})


def get_cue_epochs(raw, events, tmin=-.4, tmax=1.1, baseline=(-.4, -.1)):
    """
    Extract epochs around the cue
    :return:
    """
    event_dict = {"cue_R_reg": 112, "cue_L_reg": 111, "cue_N": 123}
    return mne.Epochs(raw, events, event_dict, tmin=tmin, tmax=tmax, baseline=baseline,
                      preload=True)  # , reject={'eeg': 400e-6})


def get_prime_cue_combination_trials(cue_epochs: mne.Epochs, prime_epochs: mne.Epochs) -> Tuple[
    np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    To avoid using cue information in prime decoding, we need to split to left
    (right) cue trials, and tag each trial with the correct prime identity
    :param cue_epochs:
    :param prime_epochs:
    :return:
    """
    rr_trials = (cue_epochs.events[:, 2] == 112) & (prime_epochs.events[:, 2] == 12)
    ll_trials = (cue_epochs.events[:, 2] == 111) & (prime_epochs.events[:, 2] == 11)
    rl_trials = (cue_epochs.events[:, 2] == 111) & (prime_epochs.events[:, 2] == 12)
    lr_trials = (cue_epochs.events[:, 2] == 112) & (prime_epochs.events[:, 2] == 11)
    rn_trials = (cue_epochs.events[:, 2] == 123) & (prime_epochs.events[:, 2] == 22)
    ln_trials = (cue_epochs.events[:, 2] == 123) & (prime_epochs.events[:, 2] == 21)
    return rr_trials, ll_trials, rl_trials, lr_trials, rn_trials, ln_trials


def get_trial_pip_score(args) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Calculate the pip score for a specific trial.
    Compatible with multiprocessing.Pool.map function
    :param args: Should contain:
        * epochs_data - np.array of shape (n_trials, n_electrodes, n_timepoints)
            containing the electrode voltages
        * labels - np.array of shape (n_trials,)
            containing the labels of the trials (0-incongruent, 1-congruent)
        * epoch_vec - np.array of shape (n_trials,)
            with all True values, is used for splitting to test and train
        * trial_num - the number of the trial to calculate PIP score for
    :return: Variants of the pip score and probability ratio over timepoints
    """
    epochs_data, labels, epoch_vec, trial_num = args
    lda = LinearDiscriminantAnalysis()
    epoch_vec[trial_num] = False
    train_x, train_y = epochs_data[epoch_vec, ...], labels[epoch_vec]
    test_x, test_y = epochs_data[~epoch_vec, ...], labels[~epoch_vec]
    probs = np.zeros((epochs_data.shape[2], 2), dtype=np.float32)
    # create classifier for each timepoint
    for timepoint in range(epochs_data.shape[2]):
        lda.fit(train_x[:, :, timepoint], train_y)
        probs[timepoint, :] = lda.predict_proba(test_x[:, :, timepoint])
    probs_ratio = probs[:, test_y] / probs[:, 1 - test_y]  # correct / incorrect
    # reset the boolean vector
    epoch_vec[trial_num] = True
    return np.mean(probs_ratio), np.median(probs_ratio), np.mean(probs[:, test_y]), np.median(
        probs[:, test_y]), probs_ratio


def get_trial_pip_score_sliding(args) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Calculate the pip score for a specific trial.
    Compatible with multiprocessing.Pool.map function
    :param args: Should contain:
        * epochs_data - np.array of shape (n_trials, n_electrodes, n_timepoints)
            containing the electrode voltages
        * labels - np.array of shape (n_trials,)
            containing the labels of the trials (0-incongruent, 1-congruent)
        * epoch_vec - np.array of shape (n_trials,)
            with all True values, is used for splitting to test and train
        * trial_num - the number of the trial to calculate PIP score for
    :return: Variants of the pip score and probability ratio over timepoints
    """
    epochs_data, labels, epoch_vec, trial_num = args
    wanted_n_windows = 10
    winsize = epochs_data.shape[2] // wanted_n_windows
    lda = LinearDiscriminantAnalysis()
    epoch_vec[trial_num] = False
    train_x, train_y = epochs_data[epoch_vec, ...], labels[epoch_vec]
    test_x, test_y = epochs_data[~epoch_vec, ...], labels[~epoch_vec]
    n_windows = np.ceil(epochs_data.shape[2] / winsize).astype(int)
    probs = np.zeros((n_windows, 2), dtype=np.float32)
    # create classifier for each timepoint
    for win_idx in range(wanted_n_windows):
        cur_train_win = train_x[:, :, win_idx * winsize:((win_idx + 1) * winsize)]
        cur_test_win = test_x[:, :, win_idx * winsize:((win_idx + 1) * winsize)]
        cur_train_x = cur_train_win.reshape((cur_train_win.shape[0], cur_train_win.shape[1] * cur_train_win.shape[2]))
        cur_test_x = cur_test_win.reshape((cur_test_win.shape[0], cur_test_win.shape[1] * cur_test_win.shape[2]))
        lda.fit(cur_train_x, train_y)
        probs[win_idx, :] = lda.predict_proba(cur_test_x)
    probs_ratio = probs[:, test_y] / probs[:, 1 - test_y]  # correct / incorrect
    # reset the boolean vector
    epoch_vec[trial_num] = True
    return np.mean(probs_ratio), np.median(probs_ratio), np.mean(probs[:, test_y]), np.median(
        probs[:, test_y]), probs_ratio


def prepare_data_for_pip_scoring(epochs,sfreq):
    """
    generate the relevant variables for "get_trial_pip_score" - scale the data, take only the relevant times
    :param epochs: The epochs to prepare
    :return:
    """
    epochs_data = epochs.get_data('eeg').astype(np.float32)
    epochs_vec = np.ones(epochs_data.shape[0], dtype=bool)
    zero_idx = np.where(epochs.times == 0)[0].item()
    min_time = 0.1
    max_time = 0.6  # s
    min_idx = np.round(sfreq * min_time).astype(int)
    max_idx = np.round(sfreq* max_time).astype(int)
    epochs_data = epochs_data[:, :, zero_idx + min_idx:zero_idx + max_idx]
    for i in range(epochs_data.shape[2]):
        epochs_data[:, :, i] = StandardScaler().fit_transform(epochs_data[:, :, i])
    return epochs_data, epochs_vec


def get_condition_prime_epochs_and_labels(prime_epochs: mne.Epochs, cong_trials: np.array, incong_trials: np.array) -> \
        Tuple[np.array, np.array]:
    """
    create prime epochs and labels for the given condition
    :param prime_epochs: Epochs of all trials locked to prime onset
    :param cong_trials: boolean vector of the congruent trials for this condition
    :param incong_trials: boolean vector of the incongruent trials for this condition
    :return: The condition epochs & labels
    """
    labels = prime_epochs.events[:, 2].copy()
    labels[incong_trials] = 0  # set rl trials to class 0
    labels[cong_trials] = 1  # set ll trials to class 1
    prime_epochs_c = prime_epochs[(labels == 0) | (labels == 1)]
    labels_c = labels[(labels == 0) | (labels == 1)]
    return prime_epochs_c, labels_c


def get_behavioral_df(s, raw, events, prime_epochs_lc, prime_epochs_rc, prime_epochs_nc):
    # calculate RT
    lc_rt = raw.times[events[prime_epochs_lc.selection + 2, 0]] - raw.times[events[prime_epochs_lc.selection + 1, 0]]
    rc_rt = raw.times[events[prime_epochs_rc.selection + 2, 0]] - raw.times[events[prime_epochs_rc.selection + 1, 0]]
    nc_rt = raw.times[events[prime_epochs_nc.selection + 2, 0]] - raw.times[events[prime_epochs_nc.selection + 1, 0]]

    # pre-allocate the rows and columns
    df = pd.DataFrame(np.zeros((lc_rt.size + rc_rt.size + nc_rt.size, 11)),
                      columns=['subject', 'prime', 'cue', 'choice', 'congruent', 'is_correct', 'RT', 'mean_ratio_pip',
                               'median_ratio_pip', 'mean_pip', 'median_pip'])
    # fill data
    df['subject'] = s
    # prime_epochs_lc.selection contains the indices in the events array for the epochs used after rejection
    df['prime'] = np.hstack([events[prime_epochs_lc.selection, 2], events[prime_epochs_rc.selection, 2],
                             events[prime_epochs_nc.selection, 2]])
    df['prime'].replace({12: "r", 11: "l", 22: "r", 21: "l"}, inplace=True)
    df['cue'] = np.hstack([events[prime_epochs_lc.selection + 1, 2], events[prime_epochs_rc.selection + 1, 2],
                           events[prime_epochs_nc.selection + 1, 2]])
    df['cue'].replace({112: "r", 111: "l", 123: "fc"}, inplace=True)
    df['choice'] = np.hstack([events[prime_epochs_lc.selection + 2, 2], events[prime_epochs_rc.selection + 2, 2],
                              events[prime_epochs_nc.selection + 2, 2]])
    df['choice'].replace({212: "r", 211: "l", 222: "r", 221: "l"}, inplace=True)
    df['congruent'] = (df['prime'] == df['cue']).astype(int)
    df['is_correct'] = (df['cue'] == df['choice']).astype(int)
    df['RT'] = np.hstack([lc_rt, rc_rt, nc_rt])
    return df


def plot_pip_rt_regression(df, type_list):
    n_types = len(type_list)
    nrow = np.round(np.sqrt(n_types)).astype(int)
    ncol = np.ceil(n_types / nrow).astype(int)
    fig = plt.figure()
    axes = fig.subplots(nrow, ncol)
    for i in range(n_types):
        name: str = type_list[i]
        q = df[f"{name}_pip"].quantile(0.95)
        sns.regplot(f"{name}_pip", "RT", df[(df[f"{name}_pip"] < q) & (df['congruent'] == 0)],
                    ax=axes[i // ncol, i % ncol],
                    color="darkred", scatter_kws={"alpha": 0.3})
        sns.regplot(f"{name}_pip", "RT", df[(df[f"{name}_pip"] < q) & (df['congruent'] == 1)],
                    ax=axes[i // ncol, i % ncol],
                    color="darkblue", scatter_kws={"alpha": 0.3})
        axes[i // ncol, i % ncol].legend(["Incongruent", "Congruent"])
        axes[i // ncol, i % ncol].set_title(name[0].upper() + name[1:].replace("_", " "))
    fig.tight_layout()


def get_frequency_features(epochs):
    freqs = dict(theta=[4, 8],
                 alpha=[8, 12],
                 beta=[12, 30],
                 gamma=[30, 45])
    tfr = mne.time_frequency.psd_multitaper(epochs.copy().crop(epochs.times[0], 0), fmax=125, n_jobs=12, picks='eeg')
    power_per_ch_dict = {}
    for k in freqs.keys():
        curr_power = tfr[0][:, :, (tfr[1] > freqs[k][0]) & (tfr[1] < freqs[k][1])].mean(
            axis=2)  # get average power in each trial and channel for the requested frequency
        for ch_idx in range(len(epochs.ch_names[:64])):
            power_per_ch_dict[k + '_' + epochs.ch_names[ch_idx]] = curr_power[:, ch_idx]
    return pd.DataFrame(power_per_ch_dict)


#
# def get_beta_during_trial(epochs):
#     freqs = dict(beta=[12, 30])
#     tfr = mne.time_frequency.psd_multitaper(epochs.copy().crop(.1, .6), fmax=125, n_jobs=12, picks='eeg')
#     power_per_ch_dict = {}
#     for k in freqs.keys():
#         curr_power = tfr[0][:, :, (tfr[1] > freqs[k][0]) & (tfr[1] < freqs[k][1])].mean(
#             axis=2)  # get average power in each trial and channel for the requested frequency
#         for ch_idx in range(len(epochs.ch_names[:64])):
#             power_per_ch_dict[k + '_' + epochs.ch_names[ch_idx]] = (-1 * channel_left[ch_idx]) * curr_power[:, ch_idx]
#
#     return pd.DataFrame(power_per_ch_dict)


def get_exponent(epochs):
    import fooof as f
    fg = f.FOOOFGroup(peak_width_limits=[1, 8], min_peak_height=0.02, max_n_peaks=8)
    psd = mne.time_frequency.psd_welch(epochs.copy().crop(epochs.times[0], 0), n_fft=1400, n_per_seg=450, n_overlap=20,
                                       fmax=125, n_jobs=12, picks='eeg')
    slope_per_ch_dict = {}
    curr_power = (psd[0][:, :, (psd[1] > 2) & (psd[1] < 100)])
    curr_freqs = (psd[1][(psd[1] > 2) & (psd[1] < 100)])
    for i in tqdm(range(len(curr_power))):
        fg.fit(curr_freqs, curr_power[i], [1, 115], n_jobs=12)
        curr_exps = fg.get_params('aperiodic_params', 'exponent')
        for ch_idx in range(len(epochs.ch_names[:64])):
            slope_per_ch_dict['slope_' + epochs.ch_names[ch_idx]] = curr_exps[ch_idx]

    return pd.DataFrame(slope_per_ch_dict)


def get_binned_data(epochs, bin_dur=10):
    dat = epochs.copy().crop(epochs.times[0], 0).get_data('eeg')
    bin_start = np.arange(epochs.times[0], 0, bin_dur / 1000) * epochs.info['sfreq']
    bin_start -= bin_start[0]
    bin_start = np.array(bin_start, dtype=int)
    dat_mov_avg = np.array(
        [dat[:, :, bin_start[curr_t]:bin_start[curr_t + 1]].mean(axis=2) for curr_t in range(len(bin_start) - 1)])
    # dat_mov_avg = np.swapaxes(np.swapaxes(dat_mov_avg,1,0),2,1)
    binned_df = {}
    for k in range(len(bin_start) - 1):
        for ch_idx in range(len(epochs.ch_names[:64])):
            binned_df[epochs.ch_names[ch_idx] + '_bin_' + str(k)] = dat_mov_avg[k, :, ch_idx] - np.mean(
                dat_mov_avg[k, :, ch_idx])
    return pd.DataFrame(binned_df)


def get_filtered_data(epochs, freq=10):
    epochs = epochs.filter(l_freq=freq - 1.5, h_freq=freq + 1.5)
    dat = epochs.copy().crop(-.125, 0).get_data('eeg')
    dat = dat[:, :, ::15]
    binned_df = {}
    for k in range(dat.shape[2] - 1):
        for ch_idx in range(dat.shape[1]):
            binned_df[epochs.ch_names[ch_idx] + '_bin_' + str(k)] = dat[:, ch_idx, k] - np.mean(
                dat[:, ch_idx, k])
    return pd.DataFrame(binned_df)


def calculate_pip_scores(data, num):
    global pool
    lc_pip_scores = []
    lc_pip_probs = []
    for res in tqdm(pool.imap(get_trial_pip_score, data), total=num):
        lc_pip_scores.append(res[:4])
        lc_pip_probs.append(res[-1])
    return np.array(lc_pip_scores), np.squeeze(np.array(lc_pip_probs))
