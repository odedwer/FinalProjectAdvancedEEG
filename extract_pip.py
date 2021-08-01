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

import utils

# %% prepare data for pip extraction
if __name__ == "__main__":
    subjects = [331, 332, 333, 334, 335, 336, 337, 339, 342, 343, 344, 345, 346, 347, 349, 349]
    subjects = [339]
    failed_subject = []
    for s in subjects:
        try:
            print(f"Reading data for subject {s}...")
            raw = utils.load_data_clean(s)
            event_dict_reg = {"prime_R_reg": 12, "prime_L_reg": 11}
            events = mne.find_events(raw, mask=255)

            # do incongruent VS congruent decoding on right and left cues
            prime_epochs = utils.get_prime_epochs(raw, events)
            cue_epochs = utils.get_cue_epochs(raw, events)

            rr_trials, ll_trials, rl_trials, lr_trials, rn_trials, ln_trials = utils.get_prime_cue_combination_trials(
                cue_epochs, prime_epochs)
            # extract epochs for trials where the cue was left/right
            prime_epochs_lc, labels_lc = utils.get_condition_prime_epochs_and_labels(prime_epochs, ll_trials,
                                                                                     rl_trials)  # type: mne.Epochs,np.array
            prime_epochs_rc, labels_rc = utils.get_condition_prime_epochs_and_labels(prime_epochs, rr_trials, lr_trials)
            prime_epochs_nc, labels_nc = utils.get_condition_prime_epochs_and_labels(prime_epochs, rn_trials, ln_trials)
            epochs_data_lc, epoch_vec_lc = utils.prepare_data_for_pip_scoring(prime_epochs_lc, raw.info['sfreq'])
            epochs_data_rc, epoch_vec_rc = utils.prepare_data_for_pip_scoring(prime_epochs_rc, raw.info['sfreq'])
            epochs_data_nc, epoch_vec_nc = utils.prepare_data_for_pip_scoring(prime_epochs_nc, raw.info['sfreq'])
            pool = mp.Pool()
            lc_pool_data = [(epochs_data_lc, labels_lc, epoch_vec_lc, i) for i in range(epochs_data_lc.shape[0])]
            rc_pool_data = [(epochs_data_rc, labels_rc, epoch_vec_rc, i) for i in range(epochs_data_rc.shape[0])]
            nc_pool_data = [(epochs_data_nc, labels_nc, epoch_vec_nc, i) for i in range(epochs_data_nc.shape[0])]

            print("Calculating left condition scores...")
            lc_pip_scores, lc_pip_probs = utils.calculate_pip_scores(lc_pool_data, epochs_data_lc.shape[0])
            print("Calculating right condition scores...")
            rc_pip_scores, rc_pip_probs = utils.calculate_pip_scores(rc_pool_data, epochs_data_rc.shape[0])
            print("Calculating free choice condition scores...")
            nc_pip_scores, nc_pip_probs = utils.calculate_pip_scores(nc_pool_data, epochs_data_nc.shape[0])
            # extract the behavioral data
            df = utils.get_behavioral_df(s, raw, events, prime_epochs_lc, prime_epochs_rc, prime_epochs_nc)
            # add the pip scores
            df['mean_ratio_pip'] = np.hstack([lc_pip_scores[:, 0], rc_pip_scores[:, 0], nc_pip_scores[:, 0]])
            df['median_ratio_pip'] = np.hstack([lc_pip_scores[:, 1], rc_pip_scores[:, 1], nc_pip_scores[:, 1]])
            df['mean_pip'] = np.hstack([lc_pip_scores[:, 2], rc_pip_scores[:, 2], nc_pip_scores[:, 2]])
            df['median_pip'] = np.hstack([lc_pip_scores[:, 3], rc_pip_scores[:, 3], nc_pip_scores[:, 3]])
            # plot regression
            df = pd.concat([df, utils.get_frequency_features(
                mne.concatenate_epochs([prime_epochs_lc, prime_epochs_rc, prime_epochs_nc]))],
                           axis=1)
            utils.plot_pip_rt_regression(df, ["median_ratio", "mean_ratio", "median", "mean"])
            plt.savefig(f"S{s}_pip_rt_reg.png")
            df.to_csv(f"S{s}_df.csv")
            np.save(f"S{s}_lc_prob_ratios.npy", lc_pip_probs)
            np.save(f"S{s}_rc_prob_ratios.npy", rc_pip_probs)
            np.save(f"S{s}_rc_prob_ratios.npy", nc_pip_probs)
        except Exception as e:
            failed_subject.append(s)
            print(f"==================================================\n"
                  f"==================================================\n"
                  f"An error has occurred in subject {s}! The exception is:\n\n{str(e)}\n\n"
                  f"==================================================\n"
                  f"==================================================\n", file=stderr)
    print(failed_subject)
