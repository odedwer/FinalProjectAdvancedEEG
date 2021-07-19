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
def load_data(subject_number, eog_list=['Nose', 'LHEOG', 'RHEOG', 'RVEOGI'],
              emg_list=['LeftEMGprox', 'rightEMGprox', 'rightEMGdist', 'LeftEMGdist']
              ) -> mne.io.Raw:
    filename = f"PVI_EEGjunior_S{subject_number}_1.bdf"
    raw: mne.io.Raw = mne.io.read_raw_bdf(filename, preload=True)
    if raw.ch_names[0][:2]=='1-':
        c=raw.ch_names[:-1]
        c=[k[2:] for k in c]
        for k in c:
            raw._orig_units[k] = raw._orig_units[f"1-{k}"]
            del raw._orig_units[f"1-{k}"]
        for k in range(len(c)):
            raw.info['chs'][k]['ch_name'] = raw.info['chs'][k]['ch_name'][2:]
            raw.ch_names[k] = c[k]

    pu.set_ch_types(raw, eog_list, 'eog')
    pu.set_ch_types(raw, emg_list, 'emg')
    raw = raw.set_montage('biosemi64')
    return raw


subject = 349
raw = load_data(subject)
raw_for_ica = raw.copy()
raw_for_ica.notch_filter([57.5, 60, 120, 180, 240], method='fir')
raw_for_ica.filter(l_freq=1, h_freq=100, method='iir')
events = mne.find_events(raw, mask=255, mask_type='and')
raw_for_ica = pu.annotate_bads_auto(raw_for_ica, 300e-6, 3000e-6)
onset_new = raw_for_ica.annotations.onset - 1
onset_new[onset_new < 0] = 0
annot_new = mne.Annotations(onset_new, np.ones_like(onset_new), raw_for_ica.annotations.description)
raw_for_ica.set_annotations(annot_new)
raw_for_ica.plot_psd(reject_by_annotation=True)
raw_for_ica.plot(duration=60, n_channels=64)

# %% ica
raw_for_ica.set_eeg_reference(ch_type='eeg')
ica = mne.preprocessing.ICA(n_components=.99, random_state=97, method='infomax', max_iter=800,
                            fit_params={'extended': True})
ica.fit(raw_for_ica, picks=['eeg', 'eog'], reject={'eeg': 400e-6})
# inspect
ica.plot_components(0)
ica.plot_properties(raw_for_ica,0)
ica.plot_sources(raw_for_ica)
# ica.plot_properties(raw_for_ica, 2)
# %% clean and save
raw.set_eeg_reference(ch_type='eeg')
ica.apply(raw)
raw.copy().set_eeg_reference().filter(l_freq=2,h_freq=100).plot()
#%%
raw.info['bads'] = raw_for_ica.info['bads']
raw.set_annotations(raw_for_ica.annotations)
raw.save(f'PVI_EEGjunior_S{subject}_1-clean-raw.fif')
