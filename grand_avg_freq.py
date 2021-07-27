import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm
import mne
import preprocess_utilities as pu


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


def plot_topo_freq_by_correct(means_df, diff_mean, freq, colnames, raw):
    means_correct = means_df.groupby('is_correct').mean()
    fig, [ax0, ax1, ax2] = plt.subplots(1, 3)
    cols = colnames[colnames[:, 0] == freq, :]
    max_val = np.max(np.array(means_correct[["".join(i) for i in cols]]))
    min_val = np.min(np.array(means_correct[["".join(i) for i in cols]]))

    X = np.array(diff_mean[["".join(i) for i in cols]])
    X = np.expand_dims(X,axis=1)
    adjacency, _ = mne.channels.find_ch_adjacency(raw.info, "eeg")
    t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_1samp_test(X,adjacency=adjacency,
                                                                                   out_type='mask',n_permutations=100)
    significant_points = np.sum(np.array(clusters)[cluster_pv<0.025],axis=0)[0]

    mne.viz.plot_topomap(means_correct.loc[1][["".join(i) for i in cols]], raw.info, axes=ax0, vmin=min_val,
                         vmax=max_val)
    mne.viz.plot_topomap(means_correct.loc[0][["".join(i) for i in cols]], raw.info, axes=ax1, vmin=min_val,
                         vmax=max_val)
    mne.viz.plot_topomap(means_correct.loc[1][["".join(i) for i in cols]] -
                         means_correct.loc[0][["".join(i) for i in cols]], raw.info, axes=ax2, mask=significant_points)
    fig.suptitle(freq)
    ax0.set_title("correct")
    ax1.set_title("incorrect")
    ax2.set_title("difference (cluster at p<0.025)")
    plt.tight_layout()

def plot_topo_freq_by_pip(means_df, diff_mean, freq, colnames, raw):
    means_high = means_df.groupby('high_pip').mean()
    fig, [ax0, ax1, ax2] = plt.subplots(1, 3)
    cols = colnames[colnames[:, 0] == freq, :]
    max_val = np.max(np.array(means_high[["".join(i) for i in cols]]))
    min_val = np.min(np.array(means_high[["".join(i) for i in cols]]))

    X = np.array(diff_mean[["".join(i) for i in cols]])
    X = np.expand_dims(X,axis=1)
    adjacency, _ = mne.channels.find_ch_adjacency(raw.info, "eeg")
    t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_1samp_test(X,adjacency=adjacency,
                                                                        out_type='mask',n_permutations=100)
    print('cluster_pv', cluster_pv)
    significant_points = np.sum(np.array(clusters)[cluster_pv<0.025],axis=0)[0]

    mne.viz.plot_topomap(means_high.loc[True][["".join(i) for i in cols]], raw.info, axes=ax0, vmin=min_val,
                         vmax=max_val)
    mne.viz.plot_topomap(means_high.loc[0][["".join(i) for i in cols]], raw.info, axes=ax1, vmin=min_val,
                         vmax=max_val)
    mne.viz.plot_topomap(means_high.loc[True][["".join(i) for i in cols]] -
                         means_high.loc[0][["".join(i) for i in cols]], raw.info, axes=ax2, mask=significant_points)
    fig.suptitle(freq)
    ax0.set_title("high pip")
    ax1.set_title("low pip")
    ax2.set_title("difference (cluster at p<0.025)")
    plt.tight_layout()



# %% read data per subject and unify them
subjects = [331, 332, 333, 334, 335, 336, 337, 339, 342, 344, 345, 346, 347, 349]
raw = load_data_clean(333)
freqs = ['theta', 'alpha', 'beta', 'gamma']
all_dfs = pd.concat([pd.read_csv(f"S{s}_df.csv") for s in subjects])
colnames = []

for ch in raw.ch_names[:64]:
    for freq in freqs:
        all_dfs[freq + '_' + ch] = np.log(all_dfs[freq + '_' + ch])
        colnames.append([freq, '_', ch])
colnames = np.array(colnames)

#%%
means_df = all_dfs.groupby(['subject', 'is_correct'], as_index=False).mean()
means_df_toplot = means_df.copy()

means_df[means_df["is_correct"] == 0] = -means_df[means_df["is_correct"] == 0]  # incong becomes negative
means_df['subject'] = np.abs(means_df['subject'])
diff_mean = means_df.groupby('subject').sum()  # actually cong-incong subtraction

plot_topo_freq_by_correct(means_df_toplot, diff_mean, 'theta', colnames, raw)
plot_topo_freq_by_correct(means_df_toplot, diff_mean, 'alpha', colnames, raw)
plot_topo_freq_by_correct(means_df_toplot, diff_mean, 'beta', colnames, raw)
plot_topo_freq_by_correct(means_df_toplot, diff_mean, 'gamma', colnames, raw)

#%%
all_dfs["high_pip"] = pd.Series(all_dfs["mean_pip"]>.5)
means_df = all_dfs.groupby(['subject','high_pip'], as_index=False).mean()
means_df_toplot = means_df.copy()

means_df[means_df["high_pip"] == False] = -means_df[means_df["high_pip"] == False]  # incong becomes negative
means_df['subject'] = np.abs(means_df['subject'])

means_df_toplot["high_pip"] = pd.Series(means_df_toplot["high_pip"],dtype='category')
diff_mean = means_df.groupby('subject').sum()  # actually cong-incong subtraction

plot_topo_freq_by_pip(means_df_toplot, diff_mean, 'theta', colnames, raw)
plot_topo_freq_by_pip(means_df_toplot, diff_mean, 'alpha', colnames, raw)
plot_topo_freq_by_pip(means_df_toplot, diff_mean, 'beta', colnames, raw)
plot_topo_freq_by_pip(means_df_toplot, diff_mean, 'gamma', colnames, raw)
