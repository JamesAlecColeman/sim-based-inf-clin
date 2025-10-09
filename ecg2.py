import numpy as np
from constants2 import *
import utils2
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def runtime_stop_condn(iter_no, iter_median_scores, window_size, stop_thresh):
    if iter_no > window_size + 1:
        twave_diff_threshold = stop_thresh
        moving_avg = np.convolve(np.diff(iter_median_scores), np.ones(window_size) / window_size, mode='same')
        abs_moving_avg = np.abs(moving_avg)
        abs_thresh = np.abs(twave_diff_threshold)
        below_threshold_indices = np.where(abs_moving_avg < abs_thresh)[0]

        if len(below_threshold_indices) == 0:
            return False, None
        else:
            i_iter_final = min(below_threshold_indices)
            return True, i_iter_final

    return False, None


def get_closest_time(times_s, chosen_time_s):
    """ Finds index of element in times_s closest to chosen_time_s

        Args:
            times_s (float list): time axis points
            chosen_time_s (float): time you wish to find on time axis

        Returns:
            float: index of chosen_time_s in times_s
    """
    times_s = np.array(times_s)
    time_diffs_s = abs(times_s - chosen_time_s)
    return np.argmin(time_diffs_s)


def match_sim_and_target_times(times_sim_s, times_target_s):
    """ Finds indices of times_target_s which can be compared to times_sim_s

        Args:
            times_sim_s (float array): simulation time axis
            times_target_s (float array): target ECG time axis

        Returns:
            target_comparison_idxs (int array): indices of times_target_s corresponding to times_sim_s time points
    """
    n_sim_times = len(times_sim_s)

    target_comparison_idxs = np.empty(n_sim_times, dtype=int)

    for i, sim_time in enumerate(times_sim_s):
        target_comparison_idxs[i] = get_closest_time(times_target_s, sim_time)

    # Sanity check to ensure point on target ECG being compared to is close enough in time
    diff_tol_s = 0.001  # 1ms tolerance
    time_diffs_s = np.abs(times_target_s[target_comparison_idxs] - times_sim_s)
    max_time_diff_s = np.max(time_diffs_s)

    if max_time_diff_s > diff_tol_s:
        print(f"{times_sim_s=}")
        print(f"{times_target_s=}")
        print(f"{time_diffs_s=}")
        raise Exception(
            f"Matching time points with target ECG more than {diff_tol_s} secs apart, {max_time_diff_s}, can also be caused by the range of target data (end of target T wave less than max repol time you tried simulating)")

    return target_comparison_idxs


def plot_ecg(all_times_s, all_leads, colors=None, labels=None, linestyles=None, axes_off=True, xlims=None, show=True, fig_no=1,
             linewidth=1.5, show_zero=False, title=False, all_not_to_plot=None, text_overlays=None, legend=True, alpha=1.0, ylabel=None,
             sharey=False, rescale_signal=1.0):
    # Original plot ECG function from arxiv preprint on 6 simulated benchmarks
    """Plot ECG signals across 12 standard leads in a 4x3 subplot layout.

    Args:
        all_times_s (list of arrays): List of time arrays (in seconds), one for each ECG to plot.
        all_leads (list of dicts): Each dict maps lead names (e.g. "I", "II", "V1", etc.) to corresponding signal arrays.
        colors (list of str, optional): Line colors for each ECG. Defaults to black.
        labels (list, optional): Labels for each ECG (used in legend).
        linestyles (list of str, optional): Line styles for each ECG. Defaults to solid lines.
        axes_off (bool, optional): Whether to hide axis lines, ticks, and labels. Default is True.
        xlims (tuple, optional): Tuple of (xmin, xmax) to set x-axis limits for all subplots.
        show (bool, optional): Whether to call `plt.show()` to display the plot. Default is True.
        fig_no (int, optional): Figure number to use for the plot. Default is 1.
        linewidth (float, optional): Width of the ECG plot lines. Default is 1.5.
        show_zero (bool, optional): Whether to draw a horizontal zero line on each subplot.
        title (str or bool, optional): Optional figure-wide title. Set to string to show, or False to skip.
        all_not_to_plot (list of lists, optional): Each sublist contains lead names not to plot for a given ECG.
        text_overlays (list of str, optional): Text strings to overlay on each subplot (one per lead).

    Returns:
        None
    """
    plt.close(fig_no)

    n_ecgs = len(all_leads)

    if all_not_to_plot is None:
        all_not_to_plot = [[] for _ in range(n_ecgs)]

    if colors is None:
        colors = ["black" for _ in range(n_ecgs)]

    if labels is None:
        labels = [i for i in range(n_ecgs)]

    if linestyles is None:
        linestyles = ["-" for _ in range(n_ecgs)]

    # Plot
    lead_names = LEAD_NAMES_12

    width_px = 850
    height_px = 750
    dpi = 100  # TODO Should be 300 for publication
    width_in = width_px / dpi
    height_in = height_px / dpi

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(width_in, height_in), dpi=dpi, num=fig_no,
                             constrained_layout=True, sharey=sharey)

    if title is not False:
        fig.suptitle(title)

    axes = axes.flatten()
    for i, ax in enumerate(axes):

        lead_name = lead_names[i]
        ax.set_title(lead_names[i])
        ax.title.set_color('gray')

        if show_zero:
            ax.axhline(0, linestyle='--', color='grey')

        if axes_off:
            ax.axis('off')
            ax.title.set_bbox(dict(facecolor='none', edgecolor='none'))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.title.set_bbox(dict(facecolor='none', edgecolor='none'))

        if text_overlays is not None:
            overlay = text_overlays[i]
        else:
            overlay = ""

        ax.text(0.85, 0.9, overlay, transform=ax.transAxes, fontsize=8, color='blue', verticalalignment='top')

        for times_s, leads, color, label, linestyle, leads_not_to_plot in zip(all_times_s, all_leads, colors, labels, linestyles, all_not_to_plot):
            if lead_name in leads and lead_name not in leads_not_to_plot:
                ax.plot(times_s, leads[lead_name] * rescale_signal, color=color, label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha)



            if xlims is not None:
                ax.set_xlim(xlims)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
    if legend:
        axes[0].legend(prop = {"size": 7})

    #plt.tight_layout()

    if show:
        plt.show()


def tie_aware_proportional_split(population_diff_scores, percent_cutoff, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    score_to_keys = defaultdict(list)  # Keys grouped by score
    for key, score in population_diff_scores.items():
        score_to_keys[score].append(key)

    unique_scores = sorted(score_to_keys.keys())
    total_keys = len(population_diff_scores)
    desired_below_count = total_keys * (percent_cutoff / 100)

    keys_below = []
    keys_above = []
    cumulative_count = 0

    for score in unique_scores:
        keys_at_score = score_to_keys[score]

        group_size = len(keys_at_score)
        next_cumulative = cumulative_count + group_size

        if next_cumulative < desired_below_count:  # Entire group below cutoff
            keys_below.extend(keys_at_score)
            cumulative_count = next_cumulative
        elif cumulative_count < desired_below_count <= next_cumulative:  # Cutoff falls within this tie group

            needed = desired_below_count - cumulative_count
            needed_int = int(round(needed))

            keys_shuffled = keys_at_score[:]
            random.shuffle(keys_shuffled)

            # Assign proportionally
            keys_below.extend(keys_shuffled[:needed_int])
            keys_above.extend(keys_shuffled[needed_int:])

            cumulative_count = next_cumulative
        else:  # Entire group above cutoff
            keys_above.extend(keys_at_score)

    # Sanity checks
    all_split_keys = set(keys_below) | set(keys_above)
    original_keys = set(population_diff_scores.keys())
    assert all_split_keys == original_keys, "Mismatch between original keys and split keys!"
    assert set(keys_below).isdisjoint(set(keys_above)), "Some keys appear in both below and above lists!"
    assert len(keys_below) + len(keys_above) == len(population_diff_scores), "Total count mismatch!"
    return keys_below, keys_above


def ten_electrodes_to_twelve_leads(electrodes):
    """ Converts 10 electrode signals into a 12-lead ECG

    Args:
        electrodes (list of arrays of floats): 10 electrode signals in order [LA, RA, LL, RL, V1, V2, V3, V4, V5, V6]

    Returns:
        leads (dict): A dictionary containing the computed 12-lead ECG signals with keys:
              "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
    """

    n_electrodes = len(electrodes)
    if n_electrodes != 10:
        raise Exception(f"Trying to compute a 12-lead ECG from {n_electrodes} electrodes")

    leads = {}
    vw = 1 / 3 * (electrodes[0] + electrodes[1] + electrodes[2])
    leads["I"] = electrodes[0] - electrodes[1]
    leads["II"] = electrodes[2] - electrodes[1]
    leads["III"] = electrodes[2] - electrodes[0]
    leads["aVR"] = 3 / 2 * (electrodes[1] - vw)
    leads["aVL"] = 3 / 2 * (electrodes[0] - vw)
    leads["aVF"] = 3 / 2 * (electrodes[2] - vw)
    leads["V1"] = electrodes[4] - vw
    leads["V2"] = electrodes[5] - vw
    leads["V3"] = electrodes[6] - vw
    leads["V4"] = electrodes[7] - vw
    leads["V5"] = electrodes[8] - vw
    leads["V6"] = electrodes[9] - vw

    return leads


def compute_adjacency_displacement(xs, ys, zs, dx, grid_dict, neighbours):
    """ Compute adjacency list for mesh cells, providing displacement vectors (di, dj, dk) for neighbour cells

    Args:
        xs, ys, zs (float arrays): mesh cell center coordinates
        dx (float): mesh spatial resolution
        grid_dict (dict): coordinate to mesh idx {(x, y, z): idx, ...}
        neighbours (int array): local neighbourhood structure (see constants.py)

    Returns:
        adjacency_list (dict): of form key (cell index): [(neighbour index, (di, dj, dk)), ...]
    """
    # TODO: Use KD trees
    neighbours = neighbours * dx

    adjacency_list = defaultdict(list)

    for x, y, z in zip(xs, ys, zs):
        idx = grid_dict[(x, y, z)]

        for di, dj, dk in neighbours:  # Check each neighbour of the mesh cell to see if it exists
            ni, nj, nk = x + di, y + dj, z + dk

            if (ni, nj, nk) in grid_dict:  # Record neighbour idx and dist from point under original point idx as key
                neighbour_idx = grid_dict[(ni, nj, nk)]
                adjacency_list[idx].append((neighbour_idx, np.array([di, dj, dk])))

    if len(adjacency_list) != len(xs):
        raise Exception("Adjacency list not of same size as mesh")

    return adjacency_list


def match_times(target_times_s, target_leads, times_s):
    """ Resamples monoalg times and leads to match the pseudo ecg times, but only if those exact times already
    exist in monoalg_times_s """

    target_leads_new = {}
    # Old unsafe version
    #new_idxes = np.concatenate([np.where(target_times_s == time)[0] for time in times_s])
    new_idxes = np.concatenate([np.where(np.isclose(target_times_s, time, atol=1e-7))[0] for time in times_s])

    target_times_s_new = target_times_s[new_idxes]

    for lead_name, lead_vals in target_leads.items():
        target_leads_new[lead_name] = target_leads[lead_name][new_idxes]

    if len(target_times_s_new) != len(times_s):
        print(f"{len(target_times_s_new)=}")
        print(f"{len(times_s)=}")
        raise Exception("Target ECG times failed to match with pseudo ECG times in match_times - float error?")

    return target_times_s_new, target_leads_new


def get_neighbour_arrays(xs, ys, zs, dx, grid_dict):
    """ Precomputes all hex mesh structural relationships to speed up gradient/pseudo ECG calculations

    Args:
        xs, ys, zs (float arrays): mesh cell center coordinates
        dx (float): mesh spatial resolution
        grid_dict (dict): coordinate to mesh idx {(x, y, z): idx, ...}

    Returns:
        neighbour_arrays, neighbour_arrays2 (dicts): precomputed neighbourhood info
    """
    n_cells = len(xs)
    neighbours = NEIGHBOURS_FACE * dx

    pos_xs, neg_xs, pos_ys, neg_ys, pos_zs, neg_zs = np.ones(n_cells, dtype=int) * -1, np.ones(n_cells,
                                                                                               dtype=int) * -1, np.ones(
        n_cells, dtype=int) * -1, np.ones(n_cells, dtype=int) * -1, np.ones(n_cells, dtype=int) * -1, np.ones(n_cells,
                                                                                                              dtype=int) * -1

    unstructured_neighbour_idxs = np.empty((n_cells, 6))

    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):

        for p, (di, dj, dk) in enumerate(neighbours):
            n_x, n_y, n_z = x + di, y + dj, z + dk

            if (n_x, n_y, n_z) in grid_dict:
                # Record presence of the neighbour in correct array

                n_idx = grid_dict[(n_x, n_y, n_z)]
                unstructured_neighbour_idxs[i, p] = n_idx

                if di > 0:
                    pos_xs[i] = n_idx
                elif di < 0:
                    neg_xs[i] = n_idx
                elif dj > 0:
                    pos_ys[i] = n_idx
                elif dj < 0:
                    neg_ys[i] = n_idx
                elif dk > 0:
                    pos_zs[i] = n_idx
                elif dk < 0:
                    neg_zs[i] = n_idx

            else:
                unstructured_neighbour_idxs[i, p] = -1  # Sentinel value

    unstructured_neighbour_idxs = np.array(unstructured_neighbour_idxs, dtype=object)

    # Count valid neighbours for each axis (x, y, z) per cell
    count_x = ((pos_xs != -1).astype(int) + (neg_xs != -1).astype(int))  # Count valid neighbours for x-direction
    count_y = ((pos_ys != -1).astype(int) + (neg_ys != -1).astype(int))  # Similarly for y-direction
    count_z = ((pos_zs != -1).astype(int) + (neg_zs != -1).astype(int))  # Similarly for z-direction

    neighbour_arrays = {}
    neighbour_arrays["pos_xs"], neighbour_arrays["neg_xs"], neighbour_arrays["pos_ys"], neighbour_arrays["neg_ys"], \
    neighbour_arrays["pos_zs"], neighbour_arrays["neg_zs"] = pos_xs, neg_xs, pos_ys, neg_ys, pos_zs, neg_zs
    neighbour_arrays["count_x"], neighbour_arrays["count_y"], neighbour_arrays["count_z"] = count_x, count_y, count_z
    neighbour_arrays["unstructured_neighbour_idxs"] = unstructured_neighbour_idxs

    original_idxs = np.array(np.arange(0, n_cells, 1), dtype=int)

    axes_arr = np.array([0, 0, 1, 1, 2, 2])  # x, y or z axis
    idxs_arr = np.array([pos_xs, neg_xs, pos_ys, neg_ys, pos_zs, neg_zs])  # mesh indices of neighbours in 6 directions
    offsets_arr = np.array([1, -1, 1, -1, 1, -1])  # positive or negative direction indicator

    valid_mask = (idxs_arr != -1)  # Denotes where neighbours actually exist (6, n_cells)
    valid_mask_flat = valid_mask.flatten()  # (6 * n_cells,)

    # Extract valid positions from the mask (valid positions in the mesh)
    # (103048,)
    valid_neighbors_idx = np.where(valid_mask_flat)[0]  # Flattened valid neighbor indices (< 6 * n_cells,)

    # Should go 0 0 0 0 0 0 0 0 0 ... 0 1 1 1 1 ...
    valid_directions = np.repeat(axes_arr, n_cells)  # Repeating directions for each neighbor (6 neighbors per cell)

    valid_offsets = np.repeat(offsets_arr, n_cells)  # Repeating offsets for each neighbor (6 neighbors per cell)

    tiled_idxs = np.tile(original_idxs, 6)  # (120066,) maximum value of n_cells

    # Now we can extract the correct direction and offset for each valid neighbor
    valid_directions_for_neighbors = valid_directions[
        valid_neighbors_idx]  # Directions corresponding to valid neighbors
    valid_offsets_for_neighbors = valid_offsets[valid_neighbors_idx]  # Offsets corresponding to valid neighbors
    valid_positions = tiled_idxs[
        valid_neighbors_idx]  # valid_neighbors_idx[tiled_idxs] # Divide by 6 because there are 6 neighbors per position

    #  vms_diff = vms[valid_idxs] - vms[valid_positions] what are valid idxs vs. valid positions
    valid_idxs = idxs_arr.flatten()[valid_mask_flat]  # Flatten idxs_arr to get the valid neighbor indices

    neighbour_arrays2 = {}
    neighbour_arrays2["valid_idxs"] = valid_idxs
    neighbour_arrays2["valid_positions"] = valid_positions
    neighbour_arrays2["valid_directions_for_neighbors"] = valid_directions_for_neighbors
    neighbour_arrays2["valid_offsets_for_neighbors"] = valid_offsets_for_neighbors

    return neighbour_arrays, neighbour_arrays2


def precompute_elec_grads(xs, ys, zs, electrodes_xyz, dx, neighbour_arrays):
    """ Precompute ∇(1/r) term for all electrodes for ECG calculation

    Args:
        xs, ys, zs (float arrays): mesh cell center coordinates
        electrodes_xyz (tuple list): electrode positions [(x1, y1, z1), ...]
        dx (float): mesh spatial resolution
        neighbour_arrays (dict): precomputed neighbourhood info

    Returns:
        elec_grads (float array): ∇(1/r) term of shape (3, n_elec, n_cells)
    """
    n_cells, n_elec = len(xs), len(electrodes_xyz)

    elec_grad_xs = np.empty((n_elec, n_cells), dtype=np.float64)
    elec_grad_ys = np.empty((n_elec, n_cells), dtype=np.float64)
    elec_grad_zs = np.empty((n_elec, n_cells), dtype=np.float64)

    for i, elec in enumerate(electrodes_xyz):
        rs_1over = 1 / np.sqrt(utils2.calc_dist_sq(xs, ys, zs, elec[0], elec[1], elec[2]))
        elec_grads = calc_grads(rs_1over, neighbour_arrays, dx)
        elec_grad_xs[i, :], elec_grad_ys[i, :], elec_grad_zs[i, :] = elec_grads[:, 0], elec_grads[:, 1], elec_grads[:,
                                                                                                         2]
    elec_grads = np.stack([elec_grad_xs, elec_grad_ys, elec_grad_zs], axis=0)
    return elec_grads


def calc_grads(vms, neighbour_arrays, dx, special_indices=None):
    """ Computes gradients of field vms

    Args:
        vms (float array): scalar field to calculate gradients of
        neighbour_arrays (dict): precomputed neighbourhood info
        dx (float): mesh spatial resolution

    Returns:
        grad (float array): ∇vms of shape (n_cells, 3)
    """
    pos_xs, neg_xs, pos_ys, neg_ys, pos_zs, neg_zs = neighbour_arrays["pos_xs"], neighbour_arrays["neg_xs"], \
    neighbour_arrays["pos_ys"], neighbour_arrays["neg_ys"], neighbour_arrays["pos_zs"], neighbour_arrays["neg_zs"]
    count_x, count_y, count_z = neighbour_arrays["count_x"], neighbour_arrays["count_y"], neighbour_arrays["count_z"]

    n_cells = len(vms)

    # Initialize gradient array (n_cells x 3) to store gradients in x, y, and z directions
    grad = np.zeros((n_cells, 3))

    # Offsets for each direction (positive and negative)
    offsets = np.array([1, -1, 1, -1, 1, -1])  # For pos/neg x, y, z directions

    # Mask for special indices, if provided
    if special_indices is not None:
        special_mask = np.zeros(n_cells, dtype=bool)
        special_mask[special_indices] = True
    else:
        special_mask = None  # Process all cells

    # Compute gradients in the x-direction
    for direction, idxs, offset in zip(np.array([0, 0, 1, 1, 2, 2]),
                                       np.array([pos_xs, neg_xs, pos_ys, neg_ys, pos_zs, neg_zs]),
                                       np.array([0, 1, 2, 3, 4, 5])):

        # Mask out invalid neighbors
        valid_mask = idxs != -1

        if special_indices is not None:
            # Compute only for special indices
            valid_special_mask = valid_mask[special_indices]
            valid_special_indices = special_indices[valid_special_mask]
            grad[valid_special_indices, direction] += (vms[idxs[valid_special_indices]] - vms[valid_special_indices]) / dx * offsets[offset]
        else:
            # Compute for all cells
            grad[valid_mask, direction] += (vms[idxs[valid_mask]] - vms[valid_mask]) / dx * offsets[offset]

    # Avoid division by zero by checking the count for each cell
    grad[:, 0] /= np.maximum(count_x, 1)  # per-cell count for x direction
    grad[:, 1] /= np.maximum(count_y, 1)  # per-cell count for y direction
    grad[:, 2] /= np.maximum(count_z, 1)  # per-cell count for z direction

    return grad


def plot_ecg_clinical_style(all_times_s, all_leads, colors=None, labels=None, linestyles=None, axes_off=True, xlims=None, show=True, fig_no=1,
             linewidth=1.5, show_zero=False, title=False, all_not_to_plot=None, text_overlays=None, legend=True, alpha=1.0, ylabel=None,
             sharey=False, rescale_signal=1.0):
    plt.rcParams['font.family'] = 'Arial'

    n_leads = len(all_leads)

    # colors, linestyles, labels
    if colors is None:
        colors = ["black" for _ in range(n_leads)]
    if linestyles is None:
        linestyles = ["-" for _ in range(n_leads)]
    if labels is None:
        labels = ["A" for _ in range(n_leads)]

    for i, leads in enumerate(all_leads):  # Apply optional rescaling to all leads
        leads_rescaled = {lead_name: leads[lead_name] * rescale_signal for lead_name in leads.keys()}
        all_leads[i] = leads_rescaled

    # ECG grid spacing (similar to 10mm/mV, 50mm/s)
    time_spacing_minor_s, time_spacing_major_s = 0.02, 0.1
    v_spacing_minor_mV, v_spacing_major_mV = 0.1, 0.5

    # Spacing of different leads
    time_left_offsets = 0.2
    v_top_offsets = 2.0
    time_per_lead_s = 0.65  #
    v_per_lead_mV = 5.0

    global_min_v, global_max_v = np.inf, -np.inf  # Track ECG signal ranges (incl offsets etc)
    global_min_t, global_max_t = np.inf, -np.inf


    fig, ax = plt.subplots(figsize=(10, 8), num=fig_no)
    lead_order = [["III", "aVF", "V3", "V6"], ["II", "aVL", "V2", "V5"], ["I", "aVR", "V1", "V4"]]
    for n, (times_s, leads, cols, styles, label) in enumerate(zip(all_times_s, all_leads, colors, linestyles, labels)):
        for row_idx, row in enumerate(lead_order):
            for col_idx, lead_name in enumerate(row):
                time_offset = col_idx * time_per_lead_s
                voltage_offset = row_idx * v_per_lead_mV
                signal = leads[lead_name] + voltage_offset + v_top_offsets
                time = times_s + time_offset + time_left_offsets

                ax.plot(time, signal,
                    label=label, color=cols, linestyle=styles)

                # Updating signal min/max
                signal_min = np.min(signal)
                signal_max = np.max(signal)
                if signal_min < global_min_v:
                    global_min_v = signal_min
                if signal_max > global_max_v:
                    global_max_v = signal_max

                t_min = np.min(time)
                t_max = np.max(time)
                if t_min < global_min_t:
                    global_min_t = t_min
                if t_max > global_max_t:
                    global_max_t = t_max

                if n == 0:
                    ax.text(t_max, voltage_offset + v_top_offsets + v_spacing_major_mV, lead_name,
                            fontsize=16, verticalalignment='bottom', horizontalalignment='right', color="gray")

    # Big range for drawing the grid lines, then we zoom in using xlim, ylim
    v_min_mV, v_max_mV = -25, +25
    time_min_s, time_max_s = -1, 3

    # Major & minor gridlines (time axis)
    xticks = np.arange(time_min_s, time_max_s + time_spacing_major_s, time_spacing_major_s)
    ax.set_xticks(xticks)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(time_spacing_minor_s))

    # Major & minor gridlines (voltage axis)
    yticks = np.arange(v_min_mV, v_max_mV + v_spacing_major_mV, v_spacing_major_mV)
    ax.set_yticks(yticks)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(v_spacing_minor_mV))

    ax.grid(True, which='major', axis='both', linewidth=0.3, zorder=0)
    ax.grid(True, which='minor', axis='both', linewidth=0.1, linestyle='-', color='gray', zorder=0)

    # Set limits
    ax.set_ylim([global_min_v - v_spacing_major_mV, global_max_v + v_spacing_major_mV])
    ax.set_xlim([global_min_t - time_spacing_major_s, global_max_t + time_spacing_major_s])

    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_antialiased(False)

    # Appearance tweaks
    ax.set_aspect(time_spacing_major_s / v_spacing_major_mV)
    ax.tick_params(axis='both', which='both', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', colors='white')  # Tick marks and labels
    ax.xaxis.label.set_color('white')  # X-axis label (if used)
    ax.yaxis.label.set_color('white')  # Y-axis label (if used)

    # Mark time and voltage
    box_start_x = 2 * time_spacing_major_s
    box_start_y = 0 * v_per_lead_mV

    box_width = time_spacing_major_s  # 100 ms (0.1s)
    box_height = v_spacing_major_mV  # 0.5 mV (0.5mV)
    ax.plot([box_start_x, box_start_x + box_width], [box_start_y, box_start_y],
            color='grey', lw=2.0, solid_capstyle='butt')
    ax.plot([box_start_x, box_start_x], [box_start_y, box_start_y + box_height],
            color='grey', lw=2.0, solid_capstyle='butt')
    ax.text(box_start_x + box_width / 2, box_start_y - (box_height * 0.3),
            '100ms', ha='center', va='top', fontsize=13, color='grey')
    ax.text(box_start_x - (box_width * 0.3), box_start_y + box_height / 2,
            '0.5mV', ha='right', va='center', fontsize=13, rotation='vertical', color='grey')

    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=13, fancybox=True, framealpha=0.85, edgecolor='gray',
            borderpad=0.3, labelspacing=0.1, handlelength=0.80, handletextpad=0.5, loc='lower right')

    if title is not False:
        plt.title(title, color="pink", fontweight="bold")

    plt.tight_layout()

    if show:
        plt.show()