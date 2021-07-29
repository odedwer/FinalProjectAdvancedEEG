import numpy as np
import mne
import matplotlib.pyplot as plt
import utils
from sys import stderr

# %% load data
average_decoding_congruency = []
subject_TOI = []  # add timepoints from which we will extract the PIP
freqs = np.arange(2, 45, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
# vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [-.4, 1]  # baseline interval (in s)

subjects = [331, 332, 333, 334, 335, 336, 337, 339, 342, 343, 344, 345, 346, 347, 349, 349]
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
        _, _, _, _, rn_trials, ln_trials = utils.get_prime_cue_combination_trials(
            cue_epochs, prime_epochs)
        # extract epochs for trials where the cue was left/right
        prime_epochs_nc, labels_nc = utils.get_condition_prime_epochs_and_labels(prime_epochs, rn_trials, ln_trials)
        # create classifier

        average_decoding_congruency.append(utils.get_decoding_scores(prime_epochs_nc, labels_nc))
    except Exception as e:
        failed_subject.append(s)
        print(f"==================================================\n"
              f"==================================================\n"
              f"An error has occurred in subject {s}! The exception is:\n\n{str(e)}\n\n"
              f"==================================================\n"
              f"==================================================\n", file=stderr)

np.save('average_decoding_congruency_neutral', average_decoding_congruency)
print('failed subjects:', failed_subject, file=stderr)
# %%
mean_subjects_average_decoding_congruency = np.mean(average_decoding_congruency, axis=0)
std_subjects_average_decoding_congruency = np.std(average_decoding_congruency, axis=0)
plt.figure()
for i in range(len(average_decoding_congruency)):
    plt.plot(prime_epochs[i].times, np.convolve(average_decoding_congruency[i], np.ones(21) / 21)[10:-10],
             color='grey',
             alpha=0.1)  # moving average for better visualization
plt.plot(prime_epochs.times, mean_subjects_average_decoding_congruency, label="average of all subjects")
boot_ci = mne.stats.bootstrap_confidence_interval(np.array(average_decoding_congruency), ci=0.99)
plt.fill_between(prime_epochs.times, boot_ci[1], boot_ci[0], color='#3b7fb9', alpha=0.25, label="99% CI")
plt.hlines(0.5, prime_epochs.times.min(), prime_epochs.times.max(), linestyles=':', colors='k', zorder=7)
plt.xlabel("Time (seconds)")
plt.ylabel("ROC Area under curve")
plt.xlim(-0.25, 1.05)
plt.ylim(0.35, 0.8)
plt.legend()
plt.title(f'Incongruent VS congruent trials decoding for left cues and right cues')
print(np.mean(mean_subjects_average_decoding_congruency[(0.6 > prime_epochs.times) & (prime_epochs.times > 0.2)]))
print(np.mean(std_subjects_average_decoding_congruency[(0.6 > prime_epochs.times) & (prime_epochs.times > 0.2)]))
print(np.std(mean_subjects_average_decoding_congruency[(0.6 > prime_epochs.times) & (prime_epochs.times > 0.2)]))
print(len(average_decoding_congruency))

# # %% cross decoding
# train_y = labels[ll_trials | rr_trials]
# test_y = labels[lr_trials | rl_trials]
# train_x = epochs_prime.get_data('eeg')[rr_trials | ll_trials, :, :]
# test_x = epochs_prime.get_data('eeg')[rl_trials | lr_trials, :, :]
#
# # %%
# train_x = train_x[:, :, 225:]
# test_x = test_x[:, :, 225:]
# test_x = test_x.reshape((test_x.shape[0], test_x.shape[1] * test_x.shape[2]))
# train_x = train_x.reshape((train_x.shape[0], train_x.shape[1] * train_x.shape[2]))
#
# lda = LinearDiscriminantAnalysis()
# permutation_scores = np.zeros(2)
# for i in tqdm(range(permutation_scores.size)):
#     lda.fit(train_x, np.random.choice(train_y, train_y.size, False))
#     permutation_scores[i] = lda.score(test_x, test_y)
# lda.fit(train_x, train_y)
# scores = lda.score(test_x, test_y)
# plt.hist(permutation_scores, bins=25)
# plt.axvline(scores, color="k", linestyle=":", linewidth=3)
# print(f'accuracy is {scores}')
# print(f'subject p-value from permutation is: {np.mean(scores < permutation_scores)}')
# # %%
# plt.figure()
# plt.plot(epochs_prime.times, scores)
# plt.hlines(0.5, epochs_prime.times.min(), epochs_prime.times.max(), linestyles=':', colors='k', zorder=7)
