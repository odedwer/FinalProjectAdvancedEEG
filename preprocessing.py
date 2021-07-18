
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

subject=336
raw=load_data(subject)
raw_for_ica = raw.copy()
raw_for_ica.notch_filter([57.5,60,120,180,240], method='fir')
raw_for_ica.filter(l_freq=1, h_freq=100, method='iir')
events = mne.find_events(raw, mask=255, mask_type='and')
raw_for_ica = pu.annotate_bads_auto(raw_for_ica, 300e-6, 3000e-6)
raw_for_ica.plot_psd()
raw_for_ica.plot()
# %% ica
ica = mne.preprocessing.ICA(n_components=.99, random_state=97, method='infomax', max_iter=800,
                            fit_params={'extended': True})
ica.fit(raw_for_ica, picks=['eeg', 'eog'], reject={'eeg': 400e-6})
# inspect
ica.plot_components(0)
ica.plot_sources(raw_for_ica)
# ica.plot_properties(raw_for_ica, 2)
# %% clean and save
ica.apply(raw)
raw.info['bads'] = raw_for_ica.info['bads']
raw.set_annotations(raw_for_ica.annotations)
raw.save(f'PVI_EEGjunior_S{subject}_1-clean-raw.fif')