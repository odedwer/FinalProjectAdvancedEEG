import numpy as np
import mne
import matplotlib.pyplot as plt
import utils
from sys import stderr

# %% load data
average_decoding_congruency = []
subject_TOI = []  # add timepoints from which we will extract the PIP
subjects = [331, 332, 333, 334, 335, 336, 337, 339, 342, 344, 345, 346, 347, 349]
failed_subject = []

freqs = np.arange(2, 45, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
# vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [-.4, 1]  # baseline interval (in s)
weights = []
cross_decoding_scores = []
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
        # 0 - left
        prime_epochs_nc, labels_nc = utils.get_condition_prime_epochs_and_labels(prime_epochs, rn_trials, ln_trials)
        # scores_nc = utils.get_decoding_scores(prime_epochs_nc, labels_nc)
        scores_nc, weight_nc = utils.get_decoding_scores_and_weights(prime_epochs_nc, labels_nc)

        prime_epochs_lc, labels_lc = utils.get_condition_prime_epochs_and_labels(prime_epochs, ll_trials, rl_trials)
        # scores_lc = utils.get_decoding_scores(prime_epochs_nc, labels_nc)
        scores_lc, weight_lc = utils.get_decoding_scores_and_weights(prime_epochs_lc, labels_lc)

        prime_epochs_rc, labels_rc = utils.get_condition_prime_epochs_and_labels(prime_epochs, rr_trials, lr_trials)
        # scores_rc = utils.get_decoding_scores(prime_epochs_rc, labels_rc)
        scores_rc, weight_rc = utils.get_decoding_scores_and_weights(prime_epochs_rc, labels_rc)

        weights.append([weight_nc, weight_rc, weight_lc])
        cross_decoding_scores.append(
            utils.get_cross_decoding_fc_to_instructed(prime_epochs_nc, prime_epochs_lc, prime_epochs_rc,
                                                      labels_nc, labels_lc, labels_rc))

        average_decoding_congruency.append((scores_rc + scores_lc + scores_nc) / 3)
    except Exception as e:
        failed_subject.append(s)
        print(f"==================================================\n"
              f"==================================================\n"
              f"An error has occurred in subject {s}! The exception is:\n\n{str(e)}\n\n"
              f"==================================================\n"
              f"==================================================\n", file=stderr)
        raise e

np.save('average_decoding_congruency', average_decoding_congruency)
np.save('cross_decoding_scores', np.array(cross_decoding_scores))
np.save('classifier_patterns', weights)
print(failed_subject)
# %%
cross_decoding_scores = np.array(cross_decoding_scores)  # subject X cue directions X timepoints
cross_decoding_scores_left = cross_decoding_scores[:, 0, :]
cross_decoding_scores_right = cross_decoding_scores[:, 1, :]


def plot_subject_average(data, times, title, ax=None):
    data = np.array(data)
    mean = np.mean(data, axis=0)
    if not ax:
        fig = plt.figure()
        ax = fig.subplots()
    for i in range(data.shape[0]):
        ax.plot(times, np.convolve(data[i], np.ones(21) / 21)[10:-10], color='grey',
                alpha=0.2)  # moving average for better visualization
    ax.plot(times, mean, label="average of all subjects", color='green')
    boot_ci = mne.stats.bootstrap_confidence_interval(data, ci=0.95)
    ax.fill_between(times, boot_ci[1], boot_ci[0], color='darkgreen', alpha=0.25, label="95% CI")
    ax.hlines(0.5, times.min(), times.max(), linestyles=':', colors='k', zorder=7)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("ROC Area under curve")
    ax.set_xlim(-0.25, 1.05)
    ax.set_ylim(0.35, 0.8)
    ax.legend()
    ax.set_title(title)


fig = plt.figure(figsize=(6, 3))
axes = fig.subplots(1, 2)
plot_subject_average(cross_decoding_scores_left, prime_epochs_nc.times,
                     f'\n Trained on free choice and tested on instructed left cues', ax=axes[0])
plot_subject_average(cross_decoding_scores_right, prime_epochs_nc.times,
                     f'\n Trained on free choice and tested on instructed right cues', ax=axes[1])
fig.suptitle("Cross-decoding scores of the prime identity")
print(
    "Correlation between right and left decoding: " + str(np.corrcoef(cross_decoding_scores_left.mean(axis=0)[400:1300],
                                                                      cross_decoding_scores_right.mean(axis=0)[
                                                                      400:1300])[0, 1]))

joint_kwargs = dict(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
grand_weights_avg_nc = mne.grand_average([curr_s_weights[0] for curr_s_weights in weights])
topo_times = [.1, .175, .25, .325, .4, .475, .55]
grand_weights_avg_nc.plot_joint(times=topo_times, title='patterns for free choice decoder', **joint_kwargs)
plt.savefig("neutral decoder weights.png")
plt.close()
grand_weights_avg_rc = mne.grand_average([curr_s_weights[1] for curr_s_weights in weights])
grand_weights_avg_rc.plot_joint(times=topo_times, title='patterns for right cue decoder', **joint_kwargs)
plt.savefig("right decoder weights.png")
plt.close()
grand_weights_avg_lc = mne.grand_average([curr_s_weights[2] for curr_s_weights in weights])
grand_weights_avg_lc.plot_joint(times=topo_times, title='patterns for left cue decoder', **joint_kwargs)
plt.savefig("left decoder weights.png")
plt.close()

plt.figure()
plt.hlines(0,prime_epochs_nc.times[0],prime_epochs_nc.times[-1],colors='k',linestyles=':')
plt.plot(prime_epochs_nc.times, np.diag(
    np.corrcoef(grand_weights_avg_nc.data.T, grand_weights_avg_lc.data.T)[1537:, :1537]),
         label="free-choice - left cue")
plt.plot(prime_epochs_nc.times, np.diag(
    np.corrcoef(grand_weights_avg_nc.data.T, grand_weights_avg_rc.data.T)[1537:, :1537]),
         label="free-choice - right cue")
plt.plot(prime_epochs_nc.times, np.diag(
    np.corrcoef(grand_weights_avg_lc.data.T, grand_weights_avg_rc.data.T)[1537:, :1537]), label="right cue - left cue")
plt.ylabel("Running weights correlation")
plt.legend()
plt.xlabel("Time(s)")
plt.title("Figure S2. Patterns of classifiers weights diverge after 500ms")
plt.ylim(-1,1)
plt.savefig("weights correlation.png")

# # %% cross decoding
# train_y = labels[ll_trials | rr_trials]
# test_y = labels[lr_trials | rl_trials]
# train_x = prime_epochs.get_data('eeg')[rr_trials | ll_trials, :, :]
# test_x = prime_epochs.get_data('eeg')[rl_trials | lr_trials, :, :]
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
# plt.plot(prime_epochs.times, scores)
# plt.hlines(0.5, prime_epochs.times.min(), prime_epochs.times.max(), linestyles=':', colors='k', zorder=7)
