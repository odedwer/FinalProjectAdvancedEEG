import mne
import numpy as np
import matplotlib.pyplot as plt


def plot_all_channels_var(raw, max_val, threshold, remove_from_top=8) -> None:
    """
    plotting the variance by channel for selecting bad channels
    :param raw: raw file
    :param threshold: color line in this number
    :param max_val: if any of the variances is larger than this value, reduce it for visualization. in case of extreme values
    :param remove_from_top: number of channels to remove from the last one. default is 16 to not include analog and face channels
    """
    channs = range(len(raw.ch_names) - remove_from_top - 1)
    data = raw.get_data(picks=channs)
    bad_points = raw._annotations.onset
    timepoints = np.round(raw._times[np.arange(len(data[1,]))], 2)
    not_bad_points = ~np.in1d(timepoints, np.round(bad_points, 2))
    var_vec = np.var(data[channs][:, not_bad_points],
                     axis=1)  # get only the ppoints that were not annotated
    # as bad for the variance calculation!
    var_vec[var_vec > max_val] = max_val  # for visualiztions
    electrode_letter = [i[0] for i in raw.ch_names[0:(len(channs))]]
    for i in channs:  # print names of noisy electrodes
        if var_vec[i] > threshold:
            print(raw.ch_names[i])
    colors = {'A': 'brown', 'B': 'red',
              'C': 'orange', 'D': 'gold',
              'E': 'green', 'F': 'blue',
              'G': 'pink', 'H': 'black'}
    plt.bar(x=raw.ch_names[0:(len(channs))], height=var_vec,
            color=[colors[i] for i in electrode_letter])
    plt.axhline(y=threshold, color='grey')
    plt.ylabel("Variance")
    plt.xlabel("channel")
    plt.show()


def get_rejection_threshold(epochs, min_thresh=30e-6, max_thresh=600e-6, n_thresh=100):
    """

    :param epochs:
    :param eog:
    :param min_thresh:
    :param max_thresh:
    :param n_thresh:
    :return:
    """
    epochs_data = epochs.get_data(picks='eeg')  # shape (n_epoch, n_elec, time)
    peak_to_peak_voltage = np.max(epochs_data, axis=2) - np.min(epochs_data, axis=2)
    peak_to_peak_voltage = np.max(peak_to_peak_voltage, axis=1)
    thresholds = np.linspace(min_thresh, max_thresh, n_thresh)
    rejections = np.repeat(thresholds, peak_to_peak_voltage.size).reshape(peak_to_peak_voltage.shape + thresholds.shape,
                                                                          order='F')
    rejections: np.array = peak_to_peak_voltage[:, None] > rejections
    reject_sum = np.sum(rejections, axis=0)
    reject_percentage = 100 * (reject_sum / peak_to_peak_voltage.shape[0])
    pairs = list(map(np.array, zip(thresholds, reject_percentage)))
    p1, p2 = pairs[0], pairs[-1]
    norm = np.linalg.norm(p2 - p1)
    distances = np.zeros(len(pairs))
    for i, p in enumerate(pairs):
        distances[i] = np.linalg.norm(np.cross(p2 - p1, p1 - p)) / norm
    max_dist_index = np.argmax(distances)

    fig = plt.figure()
    ax: plt.Axes = fig.subplots()
    ax.plot(thresholds, reject_percentage)
    ax.set_xlabel("thresholds")
    ax.set_ylabel("% Rejected epochs")

    ax.hlines([5, 10, 15, 20, 25], 0, np.max(thresholds), linestyles=":",
              label="reject percentages - 5:25 %")
    ax.vlines(thresholds[max_dist_index], 0, np.max(reject_percentage), linestyles=":", color="red", label="elbow")
    ax.legend()
    plt.show()
    selected_thresh = {'eeg': plt.ginput(timeout=0)[0][0]}
    return selected_thresh, pairs


def annotate_bads_auto(raw, reject_criteria, jump_criteria,
                       reject_criteria_blink=200e-6) -> mne.io.Raw:
    """
    reads raw object and annotates automatically by threshold criteria - lower or higher than value.
    Also reject big jumps.
    supra-threshold areas are rejected - 50 ms to each side of event
    returns the annotated raw object and print times annotated
    :param jump_criteria: number - the minimum point to-point difference for rejection
    :param raw: raw object
    :param reject_criteria: number - threshold
    :param reject_criteria_blink: number, mark blinks in order to not reject them by mistake
    :return: annotated raw object
    """
    data = raw.get_data(picks='eeg')  # matrix size n_channels X samples
    del_arr = [raw.ch_names.index(i) for i in raw.info['bads']]
    data = np.delete(data, del_arr, 0)  # don't check bad channels
    block_end = (raw['Status'][0].astype(int) & 255) == 254
    jumps = (block_end |
             (np.abs(np.sum(np.diff(data))) > len(
                 data) * jump_criteria))  # mark large changes (mean change
    # over jump threshold)
    jumps = np.append(jumps, False)  # fix length for comparing

    # reject big jumps and threshold crossings, except the beggining and the
    # end.
    rejected_times = (np.sum(np.abs(data) > reject_criteria) == 1) & \
                     ((raw.times > 0.1) & (raw.times < max(raw.times) - 0.1))
    event_times = raw.times[rejected_times]  # collect all times of
    # rejections except first and last 100ms
    plt.plot(rejected_times)
    extralist = []
    data_eog = raw.get_data(picks='eog')
    eye_events = raw.times[sum(abs(data_eog) > reject_criteria_blink) > 0]
    plt.plot(sum(abs(data_eog) > reject_criteria_blink) > 0)
    plt.title(
        "blue-annotations before deleting eye events. orange - only eye events")
    # print("loop length:", len(event_times))
    for i in range(2, len(
            event_times)):  # don't remove adjacent time points or blinks
        if i % 300 == 0: print(i)
        if ((event_times[i] - event_times[i - 1]) < .05) | \
                (np.any(abs(event_times[
                                i] - eye_events) < .3) > 0):  ## if a blink occured 300ms before or after
            extralist.append(i)
    event_times = np.delete(event_times, extralist)
    event_times = np.append(event_times, raw.times[jumps[:-1]])  # add jumps
    onsets = event_times - 0.05
    # print("100 ms of data rejected in times:\n", onsets)
    durations = [0.1] * len(event_times)
    descriptions = ['BAD_data'] * len(event_times)
    annot = mne.Annotations(onsets, durations, descriptions,
                            orig_time=raw.info['meas_date'])
    raw.set_annotations(annot)
    return raw


def set_ch_types(raw, name_list, ch_type):
    """
    :param raw: the raw input we want to set eog channels in
    :param add_channels: names of channels we would like to add
    :return: raw file with eog channels marked
    """
    map_dict = {nm: ch_type for nm in name_list}
    raw.set_channel_types(mapping=map_dict)


def plot_evokeds_with_CI(epochs_dict, channel, colors_dict, ci='se', title=" ", vlines=None, alpha=0.05, nperm=1000,
                         cluster_thresh_t=1):
    """
    plots the evoked response with confidence interval. adds a horizontal line for significance in permutation test
    :param channel: Name of the channel to plot evoked of
    :param colors_dict:  dict of the form (name:color) with identical naming to epochs_dict
    :param nperm: int, number of permutations
    :param alpha: float, significance level
    :param cluster_thresh_t: float, t-threshold for cluster.
    :param epochs_dict: dict of the form (name:epochs), must be size 2
    :param ci: the width of CI. either 'se' - standard error, 'sd' - standard deviation.
    :param title: str
    :param vlines: a list of times at which to plot a vertical dashed line.
    :return: -
    """
    if vlines is None:
        vlines = []
    if (ci != 'se') & (ci != 'sd'):
        raise ValueError("Sorry, only se or sd")
    epochs_names = list(epochs_dict.keys())
    ch_idx = epochs_dict[epochs_names[0]].ch_names.index(channel)
    epochs_perm_A = epochs_dict[epochs_names[0]]._data[:, ch_idx]
    epochs_perm_B = epochs_dict[epochs_names[1]]._data[:, ch_idx]
    T_obs, clusters, cluster_p_values, H0 = \
        mne.stats.permutation_cluster_test([epochs_perm_A, epochs_perm_B],
                                           n_permutations=nperm, seed=1,
                                           threshold=cluster_thresh_t, tail=1,
                                           out_type='mask', verbose='ERROR')
    times = epochs_dict[epochs_names[0]].times
    plt.figure()
    plt.axhline(0, color='black', lw=0.75)
    sig_clust = [clusters[i] for i in range(len(clusters)) if cluster_p_values[i] < alpha]
    sig_times_min = [np.min(times[curr_sig_clust]) for curr_sig_clust in sig_clust]
    sig_times_max = [np.max(times[curr_sig_clust]) for curr_sig_clust in sig_clust]
    for j in range(len(sig_times_min)):
        minsig, maxsig = epochs_dict[epochs_names[0]].time_as_index([sig_times_min[j], sig_times_max[j]])
        print(minsig, maxsig)
        plt.axhline(np.max(np.mean(epochs_perm_A, axis=0)), xmin=minsig / epochs_perm_A.shape[1],
                    xmax=maxsig / epochs_perm_A.shape[1],
                    color="black", lw=3, label=f"significant at p<{alpha} in cluster permutations")
    for vline in vlines:
        plt.axvline(vline, color='black', linestyle="--", lw=0.75)
    for name in epochs_dict.keys():
        curr_evoked = epochs_dict[name].average()
        n = epochs_dict[name]._data.shape[0]
        running_sd = epochs_dict[name]._data[:, 0].std(axis=0)
        if ci == 'se':
            running_sd /= np.sqrt(n)
        plt.plot(times, curr_evoked.data[ch_idx], color=colors_dict[name], label=name)  # plot data
        under_line = curr_evoked.data[ch_idx] - running_sd
        over_line = curr_evoked.data[ch_idx] + running_sd
        plt.fill_between(times, under_line, over_line, color=colors_dict[name], alpha=.1)  # std curves.
        plt.title(title)
        plt.legend()
        plt.ylim(-6e-6, 6e-6)
    plt.xlabel('time(s)')
    return T_obs, clusters, cluster_p_values, H0, sig_clust
