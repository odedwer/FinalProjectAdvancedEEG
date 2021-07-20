# %% imports
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy import stats
import mne
import matplotlib.pyplot as plt
import preprocess_utilities as pu
from tqdm import tqdm
import multiprocessing as mp
import seaborn as sns
from typing import Tuple, List
from sys import stderr


# %% functions


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


def get_prime_epochs(raw, events, tmin=-.4, tmax=1.1, baseline=(-.4, -.1)) -> mne.Epochs:
    """
    Extract epochs around the prime
    :return:
    """
    event_dict = {"prime_R": 12, "prime_L": 11}
    return mne.Epochs(raw, events, event_dict, tmin=tmin, tmax=tmax, baseline=baseline,
                      preload=True)  # , reject={'eeg': 400e-6})


def get_cue_epochs(raw, events, tmin=-.4, tmax=1.1, baseline=(-.4, -.1)):
    """
    Extract epochs around the cue
    :return:
    """
    event_dict = {"cue_R_reg": 112, "cue_L_reg": 111}
    return mne.Epochs(raw, events, event_dict, tmin=tmin, tmax=tmax, baseline=baseline,
                      preload=True)  # , reject={'eeg': 400e-6})


def get_prime_cue_combination_trials(cue_epochs: mne.Epochs, prime_epochs: mne.Epochs) -> Tuple[
    np.array, np.array, np.array, np.array]:
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
    return rr_trials, ll_trials, rl_trials, lr_trials


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
    probs = np.zeros((epochs_data.shape[2], 2))
    # create classifier for each timepoint
    for timepoint in range(epochs_data.shape[2]):
        lda.fit(train_x[:, :, timepoint], train_y)
        probs[timepoint, :] = lda.predict_proba(test_x[:, :, timepoint])
    probs_ratio = probs[:, test_y] / probs[:, 1 - test_y]  # correct / incorrect
    # reset the boolean vector
    epoch_vec[trial_num] = True
    return np.mean(probs_ratio), np.median(probs_ratio), np.mean(probs[:, test_y]), np.median(
        probs[:, test_y]), probs_ratio


def prepare_data_for_pip_scoring(epochs):
    """
    generate the relevant variables for "get_trial_pip_score" - scale the data, take only the relevant times
    :param epochs: The epochs to prepare
    :return:
    """
    epochs_data = epochs.get_data('eeg')
    epochs_vec = np.ones(epochs_data.shape[0], dtype=bool)
    zero_idx = np.where(epochs.times == 0)[0].item()
    min_time = 0.1
    max_time = 0.6  # s
    min_idx = np.round(raw.info['sfreq'] * min_time).astype(int)
    max_idx = np.round(raw.info['sfreq'] * max_time).astype(int)
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


def get_behavioral_df(s, raw, events, prime_epochs_lc, prime_epochs_rc):
    # calculate RT
    lc_rt = raw.times[events[prime_epochs_lc.selection + 2, 0]] - raw.times[events[prime_epochs_lc.selection + 1, 0]]
    rc_rt = raw.times[events[prime_epochs_rc.selection + 2, 0]] - raw.times[events[prime_epochs_rc.selection + 1, 0]]
    # pre-allocate the rows and columns
    df = pd.DataFrame(np.zeros((lc_rt.size + rc_rt.size, 11)),
                      columns=['subject', 'prime', 'cue', 'choice', 'congruent', 'is_correct', 'RT', 'mean_ratio_pip',
                               'median_ratio_pip', 'mean_pip', 'median_pip'])
    # fill data
    df['subject'] = s
    # prime_epochs_lc.selection contains the indices in the events array for the epochs used after rejection
    df['prime'] = np.hstack([events[prime_epochs_lc.selection, 2], events[prime_epochs_rc.selection, 2]])
    df['prime'].replace({12: "r", 11: "l"}, inplace=True)
    df['cue'] = np.hstack([events[prime_epochs_lc.selection + 1, 2], events[prime_epochs_rc.selection + 1, 2]])
    df['cue'].replace({112: "r", 111: "l"}, inplace=True)
    df['choice'] = np.hstack([events[prime_epochs_lc.selection + 2, 2], events[prime_epochs_rc.selection + 2, 2]])
    df['choice'].replace({212: "r", 211: "l"}, inplace=True)
    df['congruent'] = (df['prime'] == df['cue']).astype(int)
    df['is_correct'] = (df['cue'] == df['choice']).astype(int)
    df['RT'] = np.hstack([lc_rt, rc_rt])
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


# %% prepare data for pip extraction
if __name__ == "__main__":
    subjects = [331, 332, 333, 334, 335, 336, 337, 338, 339, 342, 343, 344, 345, 346, 347, 349, 349]
    #subjects = [332]
    failed_subject =[]
    for s in subjects:
        try:
            print(f"Reading data for subject {s}...")
            raw = load_data_clean(s)
            event_dict_reg = {"prime_R_reg": 12, "prime_L_reg": 11}
            events = mne.find_events(raw, mask=255)

            # do incongruent VS congruent decoding on right and left cues
            prime_epochs = get_prime_epochs(raw, events)
            cue_epochs = get_cue_epochs(raw, events)

            rr_trials, ll_trials, rl_trials, lr_trials = get_prime_cue_combination_trials(cue_epochs, prime_epochs)
            # extract epochs for trials where the cue was left/right
            prime_epochs_lc, labels_lc = get_condition_prime_epochs_and_labels(prime_epochs, ll_trials,
                                                                               rl_trials)  # type: mne.Epochs,np.array
            prime_epochs_rc, labels_rc = get_condition_prime_epochs_and_labels(prime_epochs, rr_trials, lr_trials)

            epochs_data_lc, epoch_vec_lc = prepare_data_for_pip_scoring(prime_epochs_lc)
            epochs_data_rc, epoch_vec_rc = prepare_data_for_pip_scoring(prime_epochs_rc)
            pool = mp.Pool()
            lc_pool_data = [(epochs_data_lc, labels_lc, epoch_vec_lc, i) for i in range(epochs_data_lc.shape[0])]
            rc_pool_data = [(epochs_data_rc, labels_rc, epoch_vec_rc, i) for i in range(epochs_data_rc.shape[0])]
            print("Calculating left condition scores...")
            lc_pip_scores = []
            lc_pip_probs = []
            for res in tqdm(pool.imap(get_trial_pip_score, lc_pool_data), total=epochs_data_lc.shape[0]):
                lc_pip_scores.append(res[:4])
                lc_pip_probs.append(res[-1])
            lc_pip_scores = np.array(lc_pip_scores)
            lc_pip_probs: np.array = np.squeeze(np.array(lc_pip_probs))
            print("Calculating right condition scores...")
            rc_pip_scores = []
            rc_pip_probs = []
            for res in tqdm(pool.imap(get_trial_pip_score, rc_pool_data), total=epochs_data_rc.shape[0]):
                rc_pip_scores.append(res[:4])
                rc_pip_probs.append(res[-1])
            rc_pip_scores = np.array(rc_pip_scores)
            rc_pip_probs: np.array = np.squeeze(np.array(rc_pip_probs))

            # extract the behavioral data
            df = get_behavioral_df(s, raw, events, prime_epochs_lc, prime_epochs_rc)
            # add the pip scores
            df['mean_ratio_pip'] = np.hstack([lc_pip_scores[:, 0], rc_pip_scores[:, 0]])
            df['median_ratio_pip'] = np.hstack([lc_pip_scores[:, 1], rc_pip_scores[:, 1]])
            df['mean_pip'] = np.hstack([lc_pip_scores[:, 2], rc_pip_scores[:, 2]])
            df['median_pip'] = np.hstack([lc_pip_scores[:, 3], rc_pip_scores[:, 3]])
            # plot regression
            plot_pip_rt_regression(df, ["median_ratio", "mean_ratio", "median", "mean"])
            plt.savefig(f"S{s}_pip_rt_reg.png")
            df.to_csv(f"S{s}_df.csv")
            np.save(f"S{s}_lc_prob_ratios.npy", lc_pip_probs)
            np.save(f"S{s}_rc_prob_ratios.npy", rc_pip_probs)
        except Exception as e:
            failed_subject.append(s)
            print(f"==================================================\n"
                  f"==================================================\n"
                  f"An error has occurred in subject {s}! The exception is:\n\n{str(e)}\n\n"
                  f"==================================================\n"
                  f"==================================================\n")
    print(failed_subject)