import os
import shutil
import numpy as np
import utils2
import random
from smoothing2 import gaussian_smoothing_fourier
import concurrent.futures
import monoalg_output_analysis2 as moa2
from constants2 import *
import hashlib
import ecg2
import copy
from scipy.sparse import csr_matrix


def prepare_target_leads(leads_target_qrs, leads_target_qrsandtwave, twave_safety_s):
    """Prepare and pad target ECG

        Args:
            leads_target_qrs (dict): Dictionary of QRS-phase ECG data with structure:
                                     {lead_name: [times_array_qrs, signal_array_qrs]}, where times are in seconds.
            leads_target_qrsandtwave (dict): Dictionary of full ECG data (QRS + T-wave) with structure:
                                             {lead_name: [times_array, signal_array]}.
            twave_safety_s (float): Duration (in seconds) to extend the target signal beyond the original
                                    clinical recording to safely cover the T-wave.

        Returns:
            leads_target (dict): Dictionary mapping lead names to zero-padded ECG signals
                                 (shape per lead: [n_timepoints]).
            times_target_s (np.ndarray): Updated array of target time points including T-wave padding (in seconds).
            activation_cutoff_s (float): Time point (in seconds) marking the end of the QRS complex,
                                         based on `leads_target_qrs`.
    """
    lead_names = list(leads_target_qrsandtwave.keys())
    times_target_s = leads_target_qrsandtwave[lead_names[0]][0]
    leads_target = {name: leads_target_qrsandtwave[name][1] for name in lead_names}
    # Handle timing of activation vs. repolarisation from target leads
    times_qrs_s = leads_target_qrs[lead_names[0]][0]
    activation_cutoff_s = max(times_qrs_s)

    # Setting up T wave padding
    dts_clin = np.diff(times_target_s)
    if not np.allclose(dts_clin, dts_clin[0], atol=1e-7):
        raise Exception(f"Inconsistent time steps: {dts_clin}")
    iter_dt_clin_s = dts_clin[0]
    max_t_clin_s = np.max(times_target_s)
    times_target_s = list(times_target_s)
    n_pre_safety = len(times_target_s)
    for t_clin in np.arange(max_t_clin_s + iter_dt_clin_s, max_t_clin_s + twave_safety_s, iter_dt_clin_s):
        times_target_s.append(t_clin)
    times_target_s = np.array(times_target_s)

    n_post_safety = len(times_target_s)

    for lead_name in leads_target.keys():
        for _ in range(n_post_safety - n_pre_safety):
            leads_target[lead_name] = np.append(leads_target[lead_name], 0.0)

    return leads_target, times_target_s, activation_cutoff_s


def mutate_twave_2daptable_apexb(replacement_params, min_possible_apd90_ms, max_possible_apd90_ms, all_dijk_dists_cm,
                                 trans):
    """ Mutate a single repolarisation model parameters

    Args:
        replacement_params (dict): Dictionary of current model parameters with keys:
            - "apd90_base_ms" (int): Baseline APD90 value in milliseconds.
            - "ap_shape_param" (float): Action potential shape parameter (between 0 and 1).
            - "apexb_param" (float): Apex-to-base modulation parameter (between -0.25 and 0.25).
            - "apd90_disks" (list of dicts): Local APD modulation disks, each with keys:
                - "idx": Central cell index (int).
                - "rad": Radius in cm (float).
                - "mult": APD multiplier (float).
                - "trans": Transmurality extent (0–1, float).
                - "endoepi": Side of the wall (0 = endo, 1 = epi).
        min_possible_apd90_ms (int): Lower bound on permissible APD90 value.
        max_possible_apd90_ms (int): Upper bound on permissible APD90 value.
        all_dijk_dists_cm (np.ndarray): Precomputed cell-to-cell distance matrix (shape: [n_cells, n_cells], in cm).
        trans (np.ndarray): Transmural coordinate array of each cell (shape: [n_cells], values in [0,1]).

    Returns:
        replacement_params_new (dict): New parameter dictionary with possibly mutated:
            - "apd90_base_ms" (int): Updated baseline APD90.
            - "ap_shape_param" (float): Updated AP shape parameter.
            - "apexb_param" (float): Updated apex-to-base gradient.
            - "apd90_disks" (list of frozenset): Updated list of local APD disk mutations.
    """
    # Mutation constants
    shape_param_snapping = 0.05
    p_change_ap_shapes = 0.2
    shape_param_perturb_amount = 0.20
    ap_shape_min, ap_shape_max = 0.0, 1.0

    apexb_param_snapping = 0.01
    p_change_apexb = 0.3
    apexb_perturb_amount = 0.10
    apexb_param_min, apexb_param_max = -0.25, 0.25

    min_rad_cm, max_rad_cm = 1.0, 4.0
    min_trans_amount, max_trans_amount = 0.3, 0.7
    p_endo_epi = 0.5

    p_transmurality = 0.25
    p_entire_ventricles_trans = 0.5

    p_explore = 0.3
    n_apd_mutations_explore = 10
    n_apd_mutations_exploit = 3

    min_apd_mult_explore, max_apd_mult_explore = 0.7, 1.3  # 1.3 * 300ms = 390ms
    min_apd_mult_exploit, max_apd_mult_exploit = 0.92, 1.08  # 1.08 * 300ms = 324ms

    p_new_disk = 0.5
    max_rad_disk_move_cm = 0.5
    min_rad_perturb_mult, max_rad_perturb_mult = 0.9, 1.1
    min_apd_perturb_mult, max_apd_perturb_mult = 0.95, 1.05
    min_trans_perturb_mult, max_trans_perturb_mult = 0.8, 1.2

    p_change_base_apd = 0.2

    global_rad_cm = 20.0

    n_cells = len(trans)

    apd90_base_ms = replacement_params["apd90_base_ms"]
    apd90_disks = list(replacement_params["apd90_disks"])
    apd90_disks_new = apd90_disks.copy()
    ap_shape_param = replacement_params["ap_shape_param"]

    # AP shape handling
    ap_shape_perturbation = np.arange(-shape_param_perturb_amount, shape_param_perturb_amount + shape_param_snapping, shape_param_snapping)
    ap_shape_perturbation = ap_shape_perturbation[~np.isclose(ap_shape_perturbation, 0.0)]

    if random.random() <= p_change_ap_shapes:
        possible_new_ap_shapes = ap_shape_perturbation + ap_shape_param
        possible_new_ap_shapes = possible_new_ap_shapes[(possible_new_ap_shapes >= ap_shape_min) & (possible_new_ap_shapes <= ap_shape_max)]
        ap_shape_param_new = random.choice(possible_new_ap_shapes)
        ap_shape_param_new = np.round(ap_shape_param_new / shape_param_snapping) * shape_param_snapping
    else:
        ap_shape_param_new = ap_shape_param

    # apexb handling
    apexb_param = replacement_params["apexb_param"]
    apexb_perturbation = np.arange(-apexb_perturb_amount, apexb_perturb_amount + apexb_param_snapping, apexb_param_snapping)
    apexb_perturbation = apexb_perturbation[~np.isclose(apexb_perturbation, 0.0)]

    if random.random() <= p_change_apexb:
        possible_new_apexb_params = apexb_perturbation + apexb_param
        possible_new_apexb_params = possible_new_apexb_params[
            (possible_new_apexb_params >= apexb_param_min) & (possible_new_apexb_params <= apexb_param_max)]
        apexb_param_new = random.choice(possible_new_apexb_params)
        apexb_param_new = np.round(apexb_param_new / apexb_param_snapping) * apexb_param_snapping
    else:
        apexb_param_new = apexb_param

    if random.random() <= p_change_base_apd:
        # Mutate base APD param
        base_apd_perturb = np.random.uniform(min_apd_perturb_mult, max_apd_perturb_mult)
        apd90_base_new_ms = apd90_base_ms * base_apd_perturb
        apd90_base_new_ms = np.clip(apd90_base_new_ms, min_possible_apd90_ms, max_possible_apd90_ms)
        apd90_base_new_ms = int(round(apd90_base_new_ms))

    else:
        apd90_base_new_ms = apd90_base_ms

    # Disk perturbations handling
    if random.random() <= p_explore:
        min_apd_mult, max_apd_mult = min_apd_mult_explore, max_apd_mult_explore  # Explore mutation
        n_apd_mutations = n_apd_mutations_explore
    else:
        min_apd_mult, max_apd_mult = min_apd_mult_exploit, max_apd_mult_exploit  # Exploit mutation
        n_apd_mutations = n_apd_mutations_exploit

    for _ in range(n_apd_mutations):

        # Code to add new apd90 disk perturbations
        if random.random() <= p_new_disk or len(apd90_disks) == 0:
            idx_perturb = np.random.randint(0, n_cells)
            rad_perturb_cm = np.random.uniform(min_rad_cm, max_rad_cm)
            trans_perturb = 1.0
            endoepi_perturb = 1

            if random.random() <= p_transmurality:
                if random.random() <= p_entire_ventricles_trans:
                    # Then just apply the transmural mutation to the entire ventricles, not just locally
                    rad_perturb_cm = global_rad_cm

                rand_trans_amount = np.random.uniform(min_trans_amount, max_trans_amount)
                trans_perturb = rand_trans_amount

                if random.random() <= p_endo_epi:  # Mutation from endo side
                    endoepi_perturb = 0
                else:  # Mutation from epi side
                    endoepi_perturb = 1

            mult_perturb = np.random.uniform(min_apd_mult, max_apd_mult)

            apd90_disks_new.append(frozenset({
                            "idx": idx_perturb,
                            "rad": round(rad_perturb_cm, 3),
                            "mult": round(mult_perturb, 3),
                            "trans": round(trans_perturb, 3),
                            "endoepi": endoepi_perturb
                            }.items()))

        else:  # Code to perturb existing disk perturbations

            n_existing_disks = len(apd90_disks)
            which_disk = np.random.randint(0, n_existing_disks)

            # Idx, rad, trans, mult
            disk_to_change = dict(apd90_disks[which_disk]).copy()

            # Pick new disk centre index
            dijk_dists_cm = all_dijk_dists_cm[disk_to_change["idx"]]  # All mesh cell distances from the disk centre
            idxs_near_disk_centre = np.where(dijk_dists_cm <= max_rad_disk_move_cm)[0]
            new_disk_idx = random.choice(idxs_near_disk_centre)
            disk_to_change["idx"] = new_disk_idx

            # Pick new radius
            current_rad = disk_to_change["rad"]
            if current_rad != global_rad_cm:  # Don't perturb the radius of global changes
                rad_perturb_perturb = np.random.uniform(min_rad_perturb_mult, max_rad_perturb_mult)
                new_rad = round(current_rad * rad_perturb_perturb, 3)
                new_rad = np.clip(new_rad, min_rad_cm, max_rad_cm)
            else:
                new_rad = current_rad
            disk_to_change["rad"] = new_rad

            # Pick new APD multiplier
            current_mult = disk_to_change["mult"]
            mult_perturb_perturb = np.random.uniform(min_apd_perturb_mult, max_apd_perturb_mult)
            new_mult = round(current_mult * mult_perturb_perturb, 3)
            new_mult = np.clip(new_mult, min_apd_mult_explore, max_apd_mult_explore)
            disk_to_change["mult"] = new_mult

            # Pick new transmurality
            current_trans = disk_to_change["trans"]
            if current_trans != 1.0:  # Don't perturb the transmurality of fully transmural disks
                trans_perturb_perturb = np.random.uniform(min_trans_perturb_mult, max_trans_perturb_mult)
                new_trans = round(current_trans * trans_perturb_perturb, 3)
                new_trans = np.clip(new_trans, min_trans_amount, 1.0)
            else:
                new_trans = current_trans
            disk_to_change["trans"] = new_trans

            # This works for now because any new disks are added only to the end, and we arent deleting disks
            apd90_disks_new[which_disk] = disk_to_change

    replacement_params_new = {"apd90_base_ms": apd90_base_new_ms,
                             "ap_shape_param": ap_shape_param_new,
                              "apexb_param": apexb_param_new,
                              "apd90_disks": apd90_disks_new}

    return replacement_params_new


def mutate_twave_params_2daptable_apexb(worse_keys, better_keys, all_params, min_possible_apd90_ms,
                                        max_possible_apd90_ms, all_dijk_dists_cm, trans, all_ids_and_diff_scores):
    """ Replace worse models with mutated versions of better models

    Args:
        worse_keys (list): List of keys corresponding to lower-performing individuals whose parameters will be mutated.
        better_keys (list): List of keys corresponding to higher-performing individuals used as mutation templates.
        all_params (dict): Dictionary mapping each individual key to its parameter dictionary. Each entry is expected to
                           contain APD base values, AP shape, apex-base modulation, and a list of APD disk perturbations.
        min_possible_apd90_ms (int): Minimum allowable APD90 value (in milliseconds).
        max_possible_apd90_ms (int): Maximum allowable APD90 value (in milliseconds).
        all_dijk_dists_cm (np.ndarray): Precomputed Dijkstra distances between mesh cells (shape: [n_cells, n_cells], in cm).
        trans (np.ndarray): Transmural coordinate array for each mesh cell (values between 0 and 1).
        all_ids_and_diff_scores (set): Set of previously evaluated parameter IDs (hashes), used to avoid duplicate evaluations.

    Returns:
        all_params (dict): Updated parameter dictionary with mutated values replacing those at each `worse_key`.
    """
    tried_param_ids = set()

    for i, worse_key in enumerate(worse_keys):
        replacement_apds = all_params[random.choice(better_keys)]
        n_mutation_attempts, max_mutation_attempts = 0, 1000

        while True:
            mutated_replacement_apds = mutate_twave_2daptable_apexb(replacement_apds, min_possible_apd90_ms, max_possible_apd90_ms,
                                                                    all_dijk_dists_cm, trans)
            param_id = hash_twave_param(mutated_replacement_apds)

            if param_id not in all_ids_and_diff_scores and param_id not in tried_param_ids:
                tried_param_ids.add(param_id)
                break  # Proceed with this mutated param as it is unseen

            if n_mutation_attempts > max_mutation_attempts:
                raise Exception("Failed to find a mutation that has not been tested before")
            n_mutation_attempts += 1

        all_params[worse_key] = mutated_replacement_apds

    return all_params


def hash_twave_param(param):
    """ Hashes repol model params to give an identifier

    Args:
        params (tuple tuple): repol model params

    Returns:
        string: 8 character hash identifying the activation model
    """
    h = hashlib.md5()
    # Iterate through dictionary and update hash
    for key, value in sorted(param.items()):
        h.update(key.encode('utf-8'))  #
        if isinstance(value, np.ndarray):
            h.update(value.tobytes())
        else:
            h.update(str(value).encode('utf-8'))
    # Return the first 8 characters of the hex
    return h.hexdigest()[:8]


def get_diff_score(times_sim_s, leads_twave_sim, times_target_s, leads_target, leads_qrs_sim, times_activation_s,
                   no_qrs=False):
    """ Calculate discrepancy score between simulated and target ECG T-wave signals.

    Args:
        times_sim_s (np.ndarray): Time points for simulated signals (in seconds).
        leads_twave_sim (dict): Simulated T-wave signals per lead (each a 1D array).
        times_target_s (np.ndarray): Time points for target signals (in seconds).
        leads_target (dict): Target ECG signals per lead (each a 1D array).
        leads_qrs_sim (dict): Simulated QRS segment signals per lead (each a 1D array).
        times_activation_s (np.ndarray): Activation times used to define QRS cutoff.
        no_qrs (bool, optional): If True, skip QRS-based normalization and use T-wave amplitude normalization. Default is False.

    Returns:
        tuple:
            mean_sum_abs_diffs_normapproach (float or None): Mean normalized absolute difference score across leads; None if no_qrs is True.
            times_target_aligned (np.ndarray): Target times matched to simulation times.
            sim_full_ecg_leads (dict): Concatenated rescaled QRS and T-wave simulated leads.
            leads_target (dict): The original target leads (unchanged).
    """
    lead_names = LEAD_NAMES_12

    # Limit target ECG to just the QRS
    activation_cutoff_s = max(times_activation_s)
    target_activation_idxs = np.where(times_target_s <= activation_cutoff_s)[0]

    if no_qrs:  # No QRS: normalisation will be based on the T wave ranges
        sim_qrs_amps = {name: np.max(leads_twave_sim[name]) - np.min(leads_twave_sim[name]) for name in lead_names}
        target_activation_idxs = np.arange(len(times_target_s))
    else:  # QRS: normalisation will be based on QRS range
        sim_qrs_amps = {name: np.max(leads_qrs_sim[name]) - np.min(leads_qrs_sim[name]) for name in lead_names}

    # QRS amplitudes of target and simulated leads
    target_qrs_amps = {
        name: np.max(leads_target[name][target_activation_idxs]) - np.min(leads_target[name][target_activation_idxs])
        for name in lead_names}

    # Simply scales the simulated QRS to match the target QRS in amplitude
    lambda_scaling = {name: target_qrs_amps[name] / sim_qrs_amps[name] for name in lead_names}

    leads_qrs_sim_rescaled = {name: leads_qrs_sim[name] * lambda_scaling[name] for name in lead_names}
    leads_twave_sim_rescaled = {name: leads_twave_sim[name] * lambda_scaling[name] for name in lead_names}

    # Find indices of the target times to compare each simulation time to
    target_comparison_idxs = ecg2.match_sim_and_target_times(times_sim_s, times_target_s)

    leads_twave_target = {name: leads_target[name][target_comparison_idxs] for name in lead_names}

    # For plotting
    sim_full_ecg_leads = {
        name: np.concatenate([np.array(leads_qrs_sim_rescaled[name]), np.array(leads_twave_sim_rescaled[name])]) for
        name in lead_names}

    # After flooring T wave amps (deals with flat T waves which may just be noise)
    min_amp_frac = 0.1  # 10% of QRS amplitude
    target_twave_amps = {
        name: max(
            np.max(leads_twave_target[name]) - np.min(leads_twave_target[name]),
            min_amp_frac * target_qrs_amps[name]
        )
        for name in lead_names
    }

    if not no_qrs:
        # It's the absolute difference like before, but it's scaled to the target T wave amplitude
        abs_diffs_normapproach = {
            name: np.abs(leads_twave_sim_rescaled[name] - leads_twave_target[name]) / target_twave_amps[name] for name
            in lead_names}
        sum_abs_diffs_normapproach = {name: np.mean(abs_diffs_normapproach[name]) for name in lead_names}
        mean_sum_abs_diffs_normapproach = np.mean(list(sum_abs_diffs_normapproach.values()))

        """import ecg
        # Plot what is actually being compared
        ecg2.plot_ecg([times_sim_s, times_sim_s], [leads_twave_sim_rescaled, leads_twave_target], show=True, colors=["red", "black"])
        """

    else:
        mean_sum_abs_diffs_normapproach = None

    return mean_sum_abs_diffs_normapproach, times_target_s[target_comparison_idxs], sim_full_ecg_leads, leads_target


def apd50_from_apd90(ap_shape_param, possible_apd50s):
    """
    Calculate the APD50 value corresponding to a given AP shape parameter from a predefined list.

    Args:
        ap_shape_param (float): AP shape parameter, expected to be between 0 and 1, representing
                                the relative position within the possible APD50 values.
        possible_apd50s (list or array): Sorted list or array of possible APD50 values.

    Returns:
        float: The APD50 value selected based on the scaled index from the AP shape parameter.
    """
    idx = round(ap_shape_param * (len(possible_apd50s) - 1))
    apd50 = possible_apd50s[idx]
    return apd50


def make_vms_field_2daptable(times_s, activation_times_s, twave_params, ap_table_args, repol_args_2daptable,
                             apd90_params, apd90_field_ms=None, apd50_field_ms=None):
    """ Generate voltage (Vm) field over time and space using a 2D AP table

    Args:
        times_s (np.ndarray): Array of time points (in seconds) for which Vm is computed.
        activation_times_s (np.ndarray): Array of activation times (in seconds) per cell.
        twave_params (dict): Dictionary containing parameters related to T-wave, including 'ap_shape_param'.
        ap_table_args (tuple): Tuple containing AP table data and parameters:
            (ap_table_arr, ap_table_rmps, min_apd90, max_apd90, min_apd50, max_apd50,
             apd90_step, apd50_step, ap_time_res_s, possible_apd50s_per_apd90)
        repol_args_2daptable (tuple): Tuple of parameters for repolarisation smoothing:
            (x_i, y_i, z_i, vms_grid, dx, smoothed_mask, sigma_um, smoothing_cutoff_s)
        apd90_params (np.ndarray): Array of APD90 durations (in ms) per cell.
        apd90_field_ms (np.ndarray, optional): Precomputed APD90 field (in ms) per cell. If None, computed inside.
        apd50_field_ms (np.ndarray, optional): Precomputed APD50 field (in ms) per cell. If None, computed inside.

    Returns:
        np.ndarray: 2D array of Vm values with shape (len(times_s), number_of_cells), representing the transmembrane voltage over time.
    """
    # Unpack AP table information
    (ap_table_arr, ap_table_rmps, min_apd90, max_apd90, min_apd50, max_apd50,
     apd90_step, apd50_step, ap_time_res_s, possible_apd50s_per_apd90) = ap_table_args

    # Unpack repol_args
    (x_i, y_i, z_i, vms_grid, dx, smoothed_mask, sigma_um, smoothing_cutoff_s) = repol_args_2daptable

    if apd90_field_ms is not None and apd50_field_ms is not None:
        print("Using existing AP field")  # TODO: add segmental handling
    else:

        ap_shape_param = twave_params["ap_shape_param"]

        # Construct the APD50 and APD90 field based on the parameters
        n_cells = len(activation_times_s)
        apd50_field_ms = np.empty(n_cells)

        # APD90 field simply retrieved from twave_params
        apd90_field_ms = apd90_params

        # Set cellwise APD50s from APD90s and the shape param
        for i, apd90 in enumerate(apd90_field_ms):
            apd50_field_ms[i] = apd50_from_apd90(ap_shape_param, possible_apd50s_per_apd90[apd90])


    if np.any(apd90_field_ms < 80) or np.any(apd50_field_ms < 80):
        raise ValueError("Small value in APD90/50 field: revisit how APD90/50 field is being set!")

    all_vms = np.zeros((len(times_s), len(activation_times_s)), dtype=float)  # Vms at time t

    idx_apd90_field = np.array((apd90_field_ms - min_apd90) / apd90_step, dtype=int)
    idx_apd50_field = np.array((apd50_field_ms - min_apd50) / apd50_step, dtype=int)

    # Initialise all_vms with RMPs corresponding to each APD
    for i, (idx_apd90, idx_apd50) in enumerate(zip(idx_apd90_field, idx_apd50_field)):
        all_vms[:, i] = ap_table_rmps[idx_apd90, idx_apd50]

    # Now construct Vm(t, x) based on which cells are already activated
    for i, t in enumerate(times_s):
        time_diffs = t - activation_times_s
        activated_mask = time_diffs >= 0

        # Compute the time index at each cell to access in the AP table
        time_idxs = ((time_diffs[activated_mask]) / ap_time_res_s).astype(int)
        all_vms[i][activated_mask] = ap_table_arr[
            idx_apd90_field[activated_mask], idx_apd50_field[activated_mask], time_idxs]

        # Smoothing Vms during repolarisation phase
        if t > smoothing_cutoff_s:  # Apply smoothing only during repolarisation (assumed steady state of diffusion)
            all_vms[i] = gaussian_smoothing_fourier(all_vms[i], sigma_um, x_i, y_i, z_i, vms_grid, dx, smoothed_mask)

    return all_vms


def pseudo_ecg(times_s, electrodes_xyz, elec_grads, dx, neighbour_args, all_vms, compute_grad_norms=True):
    """ Compute the pseudo ECG signals at given electrode positions from transmembrane voltages (Vm).

   Args:
       times_s (np.ndarray): Array of time points corresponding to Vm recordings.
       electrodes_xyz (np.ndarray): Coordinates of electrodes (shape: n_electrodes x 3).
       elec_grads (np.ndarray): Precomputed gradients of 1/r for each electrode and cell (shape: 3 x n_electrodes x n_cells).
       dx (float): Spatial resolution (distance between neighbouring cells).
       neighbour_args (tuple): Tuple containing neighbor data required for gradient calculation:
           (count_x, count_y, count_z, valid_idxs, valid_positions, valid_directions_for_neighbors, valid_offsets_for_neighbors)
       all_vms (np.ndarray): Vm values over time and space (shape: n_times x n_cells).
       compute_grad_norms (bool, optional): Whether to compute and return the mean gradient norm of Vm. Default is True.

   Returns:
       tuple:
           - electrodes_vs (np.ndarray): Computed pseudo ECG signals at electrodes over time (shape: n_electrodes x n_times).
           - mean_mean_grad_norms (float or None): Mean of Vm gradient norms across all cells and time steps if compute_grad_norms is True, else None.
   """
    n_elec, n_cells, n_times = len(electrodes_xyz), elec_grads.shape[2], len(times_s)
    count_x, count_y, count_z, valid_idxs, valid_positions, valid_directions_for_neighbors, valid_offsets_for_neighbors = neighbour_args
    original_idxs = np.array(np.arange(0, n_cells, 1), dtype=int)

    electrodes_vs = np.zeros((n_elec, n_times))
    mean_grad_norms = []  # Store mean grad norm across cells for each time step

    for t_idx in range(n_times):
        grad = np.zeros((n_cells, 3))
        vms = all_vms[t_idx]
        vms_diff = vms[valid_idxs] - vms[valid_positions]  # Vm difference between neighbours (6 * n_cells,)

        np.add.at(grad, (valid_positions, valid_directions_for_neighbors), (vms_diff / dx) * valid_offsets_for_neighbors)

        # Avoid division by zero by checking the count for each cell
        grad[:, 0] /= np.maximum(count_x, 1)  # per-cell count for x direction
        grad[:, 1] /= np.maximum(count_y, 1)  # per-cell count for y direction
        grad[:, 2] /= np.maximum(count_z, 1)  # per-cell count for z direction

        grad_x, grad_y, grad_z = grad[:, 0], grad[:, 1], grad[:, 2]

        # Dot ∇Vm with ∇(1/r)
        x_comp = grad_x[original_idxs].reshape(-1, 1) * elec_grads[0, :, original_idxs]
        y_comp = grad_y[original_idxs].reshape(-1, 1) * elec_grads[1, :, original_idxs]
        z_comp = grad_z[original_idxs].reshape(-1, 1) * elec_grads[2, :, original_idxs]

        # Sum the components along x, y, z for each electrode (sum over the n_cells dimension)
        electrodes_vs[:, t_idx] = -np.sum(x_comp + y_comp + z_comp, axis=0)

        # Grad norm computation is for the regularisation term in the discrepancy metric
        if compute_grad_norms:
            grad_norms = np.linalg.norm(grad, axis=1)
            mean_grad_norm = np.mean(grad_norms)  # Mean across cells in single time step
            mean_grad_norms.append(mean_grad_norm)

    if compute_grad_norms:
        mean_mean_grad_norms = np.mean(mean_grad_norms)  # Mean across cells across all time steps
    else:
        mean_mean_grad_norms = None
    return electrodes_vs, mean_mean_grad_norms


def compute_batch_ecgs(pseudo_ecg_function, times_s, electrodes_xyz, elec_grads, dx, neighbour_args, repol_args,
                       batch_indices, batch_activation_times_s, batch_twave_params, batch_apd_fields, return_vms=False,
                       ap_table_args=None, batch_all_vms={}, vms_activation=None, complete_times=None):
    """ Compute pseudo ECGs and optionally repolarisation times for a batch of simulations.

    Args:
        pseudo_ecg_function (callable): Function to compute pseudo ECG from Vm fields.
        times_s (np.ndarray): Array of time points corresponding to Vm recordings.
        electrodes_xyz (np.ndarray): Coordinates of electrodes.
        elec_grads (np.ndarray): Gradients of 1/r precomputed for each electrode and cell.
        dx (float): Spatial resolution between neighboring cells.
        neighbour_args (tuple): Neighbor information for gradient calculations.
        repol_args (tuple or None): Arguments for Vm field construction; if None, Vm fields are not constructed.
        batch_indices (iterable): Indices for the batch of simulations to process.
        batch_activation_times_s (list or dict): Activation times for each simulation in the batch.
        batch_twave_params (list or dict): T-wave parameters for each simulation.
        batch_apd_fields (list or dict): APD fields (e.g., APD90 values) for each simulation.
        return_vms (bool, optional): If True, return Vm fields alongside ECGs. Default is False.
        ap_table_args (tuple or None, optional): Arguments needed for AP table interpolation; required if repol_args is not None.
        batch_all_vms (dict, optional): Dictionary to store or reuse Vm fields per simulation index.
        vms_activation (np.ndarray or None, optional): Vm activation data used for repolarisation time calculations.
        complete_times (np.ndarray or None, optional): Time array covering both activation and repolarisation phases.

    Returns:
        tuple:
            - batch_electrodes (dict): Pseudo ECG signals for each simulation indexed by batch_indices.
            - batch_repol_times (dict): Repolarisation times per cell for each simulation; empty if repol_args is None.
            - batch_vms_to_return (dict or None): Vm fields per simulation if return_vms is True, else None.
            - batch_mean_mean_grad_norms (dict): Mean gradient norms of Vm for each simulation.
    """
    batch_electrodes, batch_repol_times, batch_mean_mean_grad_norms = {}, {}, {}

    for i_try in batch_indices:
        if repol_args is not None:  # Make Vms field
            vms_field = make_vms_field_2daptable(times_s, batch_activation_times_s[i_try], batch_twave_params[i_try],
                                                 ap_table_args, repol_args, batch_apd_fields[i_try])
            batch_all_vms[i_try] = vms_field

            if vms_activation is not None:  # Calculate post-smoothing repolarisation times

                vms_activation_and_repol = np.concatenate((vms_activation, vms_field), axis=0)

                n_cells = len(batch_activation_times_s[i_try])
                repol_times = np.empty(n_cells, dtype=float)
                none_repols = False
                none_count = 0

                for i_cell in range(n_cells):
                    apd, activation_time, repolarisation_time, ap_amp = moa2.calc_apd_s(complete_times, vms_activation_and_repol[:, i_cell],
                                                                                       activation_time=batch_activation_times_s[i_try][i_cell])
                    if repolarisation_time is None and not none_repols:  # Record failed repolarisation in the mesh
                        none_repols = True

                    if repolarisation_time is not None:
                        repol_times[i_cell] = repolarisation_time
                    else:
                        repol_times[i_cell] = 65.535  # Sentinel failed repol value (max uint16)
                        none_count +=1

                batch_repol_times[i_try] = repol_times
        # ECG calc
        batch_electrodes[i_try], batch_mean_mean_grad_norms[i_try] = pseudo_ecg_function(times_s, electrodes_xyz,
                                                                                         elec_grads, dx,
                                                                                         neighbour_args,
                                                                                         batch_all_vms[i_try])
    if return_vms:
        batch_vms_to_return = batch_all_vms
    else:
        batch_vms_to_return = None

    return batch_electrodes, batch_repol_times, batch_vms_to_return, batch_mean_mean_grad_norms


def batch_ecg_runner(n_tries, n_per_batch, pseudo_ecg_function, times_s, electrodes_xyz, elec_grads, dx, neighbour_args,
                     apd90_params, twave_params, all_activation_times_s, repol_args, ap_table_args, complete_times,
                     vms_activation, return_vms=False):
    """ Runs multiple repol ECG simulations in parallel batches with multiprocessing.

    Args:
        n_tries (int): Total number of simulation runs to perform.
        n_per_batch (int): Number of simulations to run per batch.
        pseudo_ecg_function (callable): Function to compute pseudo ECG signals.
        times_s (np.ndarray): Time points for voltage simulations.
        electrodes_xyz (np.ndarray): Electrode coordinates.
        elec_grads (np.ndarray): Precomputed gradients for electrodes.
        dx (float): Spatial resolution between cells.
        neighbour_args (tuple): Neighboring cell info for gradient computations.
        apd90_params (dict or list): APD90 parameters for each simulation.
        twave_params (dict or list): T-wave parameters for each simulation.
        all_activation_times_s (dict or list): Activation times for each simulation.
        repol_args (tuple or None): Arguments for Vm field construction.
        ap_table_args (tuple or None): Arguments needed for AP table interpolation.
        complete_times (np.ndarray): Complete time array covering activation and repolarisation.
        vms_activation (np.ndarray): Vm activation data used in repolarisation calculations.
        return_vms (bool, optional): Whether to return the Vm fields alongside ECG results.

    Returns:
        tuple:
            - all_electrodes (dict): Combined pseudo ECG signals from all batches, keyed by simulation index.
            - all_repol_times (dict): Combined repolarisation times per simulation.
            - all_vms_return (dict): Combined Vm fields if return_vms is True, else empty dict.
            - all_mean_mean_grad_norms (dict): Mean gradient norms of Vm from all simulations.
    """
    all_electrodes, all_repol_times, all_vms_return, all_mean_mean_grad_norms = {}, {}, {}, {}
    batches = [range(i, min(i + n_per_batch, n_tries)) for i in range(0, n_tries, n_per_batch)]

    batched_activation_times_s = [{} for _ in range(len(batches))]
    batched_twave_params = [{} for _ in range(len(batches))]
    batched_apd90_params = [{} for _ in range(len(batches))]

    # Precompute all activation times rather than pass in all_time_matrix to each subprocess
    for i, batch in enumerate(batches):
        for i_try in batch:

            batched_activation_times_s[i][i_try] = all_activation_times_s[i_try]
            batched_twave_params[i][i_try] = twave_params[i_try]
            batched_apd90_params[i][i_try] = apd90_params[i_try]

    # Batch multiprocess parallel execution of activation times and pseudo ECG computation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_batch_ecgs, pseudo_ecg_function, times_s, electrodes_xyz, elec_grads,
                                   dx, neighbour_args, repol_args, batch, batch_activation_times_s, batch_twave_params,
                                   batch_apd_fields, return_vms=return_vms, ap_table_args=ap_table_args,
                                   vms_activation=vms_activation, complete_times=complete_times)

                   for batch, batch_activation_times_s, batch_twave_params, batch_apd_fields in zip(batches, batched_activation_times_s,
                                                                                                    batched_twave_params, batched_apd90_params)]
        # Add electrode outputs to the storage dictionary
        for future in concurrent.futures.as_completed(futures):
            batch_electrodes, batch_repol_times, batch_vms_to_return, batch_mean_mean_grad_norms = future.result()
            all_electrodes.update(batch_electrodes)
            all_repol_times.update(batch_repol_times)
            all_mean_mean_grad_norms.update(batch_mean_mean_grad_norms)

            if batch_vms_to_return is not None:
                all_vms_return.update(batch_vms_to_return)

    return all_electrodes, all_repol_times, all_vms_return, all_mean_mean_grad_norms


def params_to_apd90s_field_apexb(params, all_dijk_dists_cm, trans, apexb, min_possible_apd90_ms, max_possible_apd90_ms):
    """ Converts T wave params to APD90 field params, from region APD multipliers, apex-to-base param and base apd90

    Args:
        params (dict): Dictionary containing T-wave parameters including:
            - "apd90_base_ms" (float): Baseline APD90 value in milliseconds.
            - "apd90_disks" (list of dicts): Each dict defines a disk region with keys:
                - "idx" (int): Index of disk center cell.
                - "rad" (float): Radius of the disk in cm.
                - "mult" (float): Multiplicative factor for APD90 in this disk.
                - "trans" (float): Transmural extent (0 to 1) of the disk region.
                - "endoepi" (int): 0 for endocardial side, 1 for epicardial side.
            - "apexb_param" (float): Parameter controlling apex-base APD90 scaling.
        all_dijk_dists_cm (np.ndarray): Matrix of shortest path distances between cells (in cm).
        trans (np.ndarray): Transmural coordinates of cells (range 0 to 1).
        apexb (np.ndarray): Apex-base normalized coordinate values per cell (range 0 to 1).
        min_possible_apd90_ms (float): Minimum allowable APD90 value (ms).
        max_possible_apd90_ms (float): Maximum allowable APD90 value (ms).

    Returns:
        np.ndarray: APD90 field (int16) per cell after regional modifications and apex-base scaling, clipped to valid range.
    """
    # Converts T wave params to apd90 field interim params
    n_cells = len(apexb)
    apd90_field_ms = np.ones(n_cells) * params["apd90_base_ms"]

    if len(params["apd90_disks"]):
        for disk in params["apd90_disks"]:  # add APD90 region multiplier effects

            disk = dict(disk)  # Cast from frozenset to dict so we can index
            dijk_dists_cm = all_dijk_dists_cm[disk["idx"]]  # All mesh cell distances from the disk centre
            idxs_in_rad = np.where(dijk_dists_cm <= disk["rad"])[0]

            if disk["trans"] <= 1.0:  # If disk is part-transmural
                trans_in_rad = trans[idxs_in_rad]
                if disk["endoepi"] == 0:  # From endo side
                    idxs_in_rad = idxs_in_rad[(trans_in_rad >= 0) & (trans_in_rad <= disk["trans"])]
                elif disk["endoepi"] == 1:  # From epi side
                    idxs_in_rad = idxs_in_rad[(trans_in_rad >= 1 - disk["trans"]) & (trans_in_rad <= 1)]

            apd90s_in_rad = apd90_field_ms[idxs_in_rad]
            new_apd90s_in_rad_temp = apd90s_in_rad * disk["mult"]
            apd90_field_ms[idxs_in_rad] = new_apd90s_in_rad_temp

    apd90_field_ms = apd90_field_ms * (1 + params["apexb_param"] * (2 * apexb - 1))  # Add the apexb param effect

    apd90_field_ms = np.clip(apd90_field_ms, min_possible_apd90_ms, max_possible_apd90_ms)  # Clip entire field after
    apd90_field_ms = apd90_field_ms.astype(np.int16)

    return apd90_field_ms


def neighbour_arrays_to_args(neighbour_arrays, neighbour_arrays2):
    """ Reorganise ECG preprocessing

    Args:
        neighbour_arrays (dict): Dictionary containing arrays for neighbor counts with keys:
            - "count_x": Count of neighbors in x-direction.
            - "count_y": Count of neighbors in y-direction.
            - "count_z": Count of neighbors in z-direction.
        neighbour_arrays2 (dict): Dictionary containing arrays related to valid neighbors with keys:
            - "valid_idxs": Indices of valid neighbors.
            - "valid_positions": Positions corresponding to valid neighbors.
            - "valid_directions_for_neighbors": Directions for neighbors.
            - "valid_offsets_for_neighbors": Offsets for neighbors.

    Returns:
        tuple: A tuple containing the following numpy arrays (all int32):
            - count_x
            - count_y
            - count_z
            - valid_idxs
            - valid_positions
            - valid_directions_for_neighbors
            - valid_offsets_for_neighbors
    """
    count_x, count_y, count_z = neighbour_arrays["count_x"].astype(np.int32), neighbour_arrays["count_y"].astype(
        np.int32), neighbour_arrays["count_z"].astype(np.int32)
    valid_idxs, valid_positions = neighbour_arrays2["valid_idxs"].astype(np.int32), neighbour_arrays2[
        "valid_positions"].astype(np.int32)
    valid_directions_for_neighbors = neighbour_arrays2["valid_directions_for_neighbors"].astype(np.int32)
    valid_offsets_for_neighbors = neighbour_arrays2["valid_offsets_for_neighbors"].astype(np.int32)

    neighbour_args = (
        count_x, count_y, count_z, valid_idxs, valid_positions, valid_directions_for_neighbors,
        valid_offsets_for_neighbors)
    return neighbour_args


def init_twave_params_apexb(n_tries, possible_apd90s_ms, n_cells, max_attempts_init=1000, ap_shape_param_init=0.5,
                      rad_perturb_cm=3.0, mult_perturb=1.05, trans_perturb=1.0, endoepi_perturb=0, apexb_init=0.0):
    """ Initialize a set of unique T-wave parameter dicts

    Args:
        n_tries (int): Number of unique parameter sets to initialize.
        possible_apd90s_ms (list or array): List of possible APD90 values (in milliseconds) to sample from.
        n_cells (int): Number of cells in the model (used for perturbation index selection).
        max_attempts_init (int, optional): Maximum number of attempts to find unique parameter sets. Defaults to 1000.
        ap_shape_param_init (float, optional): Initial AP shape parameter value. Defaults to 0.5.
        rad_perturb_cm (float, optional): Radius (in cm) of the perturbation disk. Defaults to 3.0.
        mult_perturb (float, optional): Multiplier applied within the perturbation disk. Defaults to 1.05.
        trans_perturb (float, optional): Transmurality parameter for the perturbation disk. Defaults to 1.0.
        endoepi_perturb (int, optional): Endocardial/epicardial side indicator for the perturbation disk (0 for endo, 1 for epi). Defaults to 0.
        apexb_init (float, optional): Initial apexb parameter value. Defaults to 0.0.

    Returns:
        dict: A dictionary mapping each try index (0 to n_tries-1) to its corresponding unique T-wave parameter dictionary.

    Raises:
        Exception: If unique parameter sets cannot be generated within the allowed number of attempts.
    """

    current_iter_params = {}
    params_included_already = set()
    n_attempts_init = 0

    # Initialise the population
    for i_try in range(n_tries):
        while True:
            # Try adding uniform APDs among cells
            apd90_base_ms = random.choice(possible_apd90s_ms)

            params_dict = {  # Global parameters
                "apd90_base_ms": apd90_base_ms,
                "ap_shape_param": ap_shape_param_init,
                "apexb_param": apexb_init,
                # Perturbation parameters
                "apd90_disks": tuple([])  # Tuple so it is hashable
            }

            # dicts are not hashable so we must use the frozenset form to store them
            params_dict_frozen = frozenset(params_dict.items())

            if params_dict_frozen not in params_included_already:
                # Add this APD param setup and proceed to next try
                current_iter_params[i_try] = params_dict
                params_included_already.add(params_dict_frozen)
                break
            else:
                # Perturb the configuration slightly to generate a unique one
                idx_perturb = np.random.randint(0, n_cells)  # Index of perturbation centre

                params_dict = {  # Global parameters
                    "apd90_base_ms": apd90_base_ms,
                    "ap_shape_param": ap_shape_param_init,
                    "apexb_param": apexb_init,

                    # Perturbation parameters
                    "apd90_disks": tuple([
                        frozenset({  # Nested dicts must be frozenset to be hashable
                                      "idx": idx_perturb,
                                      "rad": rad_perturb_cm,
                                      "mult": mult_perturb,
                                      "trans": trans_perturb,
                                      "endoepi": endoepi_perturb
                                  }.items())
                    ])
                }

                # dicts are not hashable so we must use the frozenset form to store them
                params_dict_frozen = frozenset(params_dict.items())

                if params_dict_frozen not in params_included_already:
                    # Add this perturbed APD param setup and proceed to next try
                    current_iter_params[i_try] = params_dict
                    params_included_already.add(params_dict_frozen)
                    break

            n_attempts_init += 1

            if n_attempts_init > max_attempts_init:
                raise Exception("Failed to initialise APDs with unique segmental APD distribution")

    # Sanity check as number of unique initial APD distributions should equal n_tries
    if len(params_included_already) != n_tries:
        raise Exception(f"Initial number of unique APD parameter sets {len(params_included_already)} != {n_tries=}")

    return current_iter_params


def monoalg_conductivity_to_smoothing_sigma(conductivity, use_grads=True):
    """ Convert a MonoAlg conductivity value to a corresponding Gaussian smoothing parameter sigma.

    This function maps MonoAlg conductivity values to smoothing sigma values based on calibration from
    a 1 cm³ simulation with specific conditions (450 ms duration, 70% endocardial, 30% epicardial cells,
    stimulation at t=0 with -53 current, spatial resolution dx=500 μm). Two calibration modes are available:
    one optimized for best raw membrane voltage (Vm) match and another optimized for best Vm gradient norm match.

    Args:
        conductivity (float): The MonoAlg conductivity value to convert.
        use_grads (bool, optional): If True, use smoothing values optimized for Vm gradient norms;
                                    if False, use values optimized for raw Vm. Defaults to True.

    Returns:
        float: Interpolated Gaussian smoothing sigma corresponding to the input conductivity.
    """

    conductivities = [0, 0.000025, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040, 0.00045,
                      0.00050, 0.00055, 0.00060]

    # Based on dx=500um slab where AP table was the original one (based on stretching/compressing time axis)
    #best_sigmas_grad_vms = [0, 1250, 2350, 2850, 3200, 3550, 3900, 4200, 4550, 4800, 5000, 5200, 5400, 5550]
    #best_sigmas_vms      = [0, 1700, 2250, 3000, 3350, 3650, 4000, 4300, 4350, 4400, 4500, 4650, 4800, 4900]

    # Based on dx=500um slab where AP table used -known- pre-smoothing epi and endo AP shapes from MonoAlg
    best_sigmas_grad_vms = [np.float64(0.0), np.float64(1400.0), np.float64(1850.0), np.float64(2600.0),
                            np.float64(3050.0), np.float64(3250.0), np.float64(3500.0), np.float64(3650.0),
                            np.float64(3900.0), np.float64(4000.0), np.float64(4150.0), np.float64(4300.0),
                            np.float64(4450.0), np.float64(4550.0)]
    best_sigmas_vms = [np.float64(0.0), np.float64(1100.0), np.float64(1850.0), np.float64(2450.0), np.float64(2850.0),
                       np.float64(3200.0), np.float64(3500.0), np.float64(3800.0), np.float64(4050.0),
                       np.float64(4350.0), np.float64(4700.0), np.float64(4900.0), np.float64(5100.0),
                       np.float64(5250.0)]

    if use_grads:
        sigmas = best_sigmas_grad_vms
    else:
        sigmas = best_sigmas_vms

    sigma_interp = utils2.linear_interpolation_arrays(conductivities, sigmas, conductivity)
    return sigma_interp


def monoalg_cv_to_conductivity(cv_cm_per_s):
    """ Convert conduction velocity (CV) in cm/s to a MonoAlg conductivity value.

    The function interpolates the conductivity corresponding to the given conduction velocity
    using predefined calibration data. Tuned based on 5cm cable 750ms duration, end cell stimulated
    at t=0 by -80 current, at dx=500um

    Args:
        cv_cm_per_s (float): Conduction velocity in centimeters per second.

    Returns:
        float: Interpolated MonoAlg conductivity value corresponding to the input CV.
    """

    conductivities = [0, 0.000025, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040, 0.00045,
                      0.00050, 0.00055, 0.00060]

    cvs_cm_per_s = [0, np.float64(7.692307692307692), np.float64(15.527950310559007), np.float64(28.571428571428573), np.float64(39.37007874015748), np.float64(48.54368932038835), np.float64(56.81818181818182), np.float64(64.1025641025641), np.float64(70.42253521126761), np.float64(76.92307692307692), np.float64(81.9672131147541), np.float64(87.71929824561403), np.float64(92.5925925925926), np.float64(96.15384615384616)]

    conductivity_interp = utils2.linear_interpolation_arrays(cvs_cm_per_s, conductivities, cv_cm_per_s)
    return conductivity_interp


def preprocess_2d_ap_table(ap_table_2d, times_sim_s, every_xth_time):
    """ Preprocess a 2D action potential (AP) table to reduce time resolution and organize data for simulation.

    This function performs the following steps:
    - Reduces the temporal resolution of each AP in the table by taking every nth time point.
    - Clips the AP data to the maximum simulation time.
    - Verifies that the APD90 and APD50 values in the table are equispaced.
    - Checks that the AP time axis is equispaced and matches the simulation times.
    - Constructs arrays of AP waveforms and resting membrane potentials (RMPs) indexed by APD90 and APD50 values.
    - Extracts metadata such as min/max APD90/APD50 values, their steps, and AP time resolution.

    Args:
        ap_table_2d (dict): Dictionary with keys as (APD90, APD50) tuples and values as [times_array, Vms_array].
        times_sim_s (np.ndarray): Array of simulation time points (in seconds).
        every_xth_time (int): Downsampling factor to reduce the AP time resolution (e.g., 5 means take every 5th point).

    Returns:
        tuple:
            ap_table_arr (np.ndarray): 3D array of shape (n_apd90s, n_apd50s, n_times_ap) containing AP voltage waveforms.
            ap_table_rmps (np.ndarray): 2D array of shape (n_apd90s, n_apd50s) containing resting membrane potentials.
            min_apd90 (float): Minimum APD90 value in the table.
            max_apd90 (float): Maximum APD90 value in the table.
            min_apd50 (float): Minimum APD50 value in the table.
            max_apd50 (float): Maximum APD50 value in the table.
            apd90_step (float): Step size between APD90 values (must be uniform).
            apd50_step (float): Step size between APD50 values (must be uniform).
            ap_time_res_s (float): Temporal resolution of the AP waveforms (in seconds).
            possible_apd50s_per_apd90 (dict): Dictionary mapping each APD90 to the list of possible APD50 values.
    """
    apd_tuples = np.array(list(ap_table_2d.keys()))
    max_sim_time = np.max(times_sim_s)

    for apd_tuple in apd_tuples:

        ts = ap_table_2d[tuple(apd_tuple)][0]
        vms = ap_table_2d[tuple(apd_tuple)][1]

        #  Reduce AP time resolution (0.2ms originally, typically we move to 1ms)
        ts_reduced = ts[::every_xth_time]
        vms_reduced = vms[::every_xth_time]

        up_to_sim_time_idxs = np.where(ts_reduced <= max_sim_time * 1.05)

        #  Reduce max AP time to the simulation time (originally 1000ms, typically goes down to about 500ms)
        ts_reduced = ts_reduced[up_to_sim_time_idxs]
        vms_reduced = vms_reduced[up_to_sim_time_idxs]

        ap_table_2d[tuple(apd_tuple)][0] = ts_reduced
        ap_table_2d[tuple(apd_tuple)][1] = vms_reduced

    ap_table_apd90s = apd_tuples[:, 0]
    ap_table_apd50s = apd_tuples[:, 1]

    min_apd90, max_apd90 = np.min(ap_table_apd90s), np.max(ap_table_apd90s)
    min_apd50, max_apd50 = np.min(ap_table_apd50s), np.max(ap_table_apd50s)

    unique_ap_table_apd90s = sorted(np.unique(ap_table_apd90s))
    d_apd90s = np.unique(np.diff(unique_ap_table_apd90s))

    if len(d_apd90s) > 1:  # Ensure AP table is equispaced in APD90
        raise Exception(f"AP table is not equispaced in APD90! Possible steps in APD90 in AP table are {d_apd90s}.")

    apd90_step = d_apd90s[0]

    possible_apd50s_per_apd90 = {}

    for apd90 in unique_ap_table_apd90s:
        # Finds all APD50s corresponding to this apd90
        requested_apd50s = apd_tuples[apd_tuples[:, 0] == apd90, 1]
        d_apd50s = np.unique(np.diff(requested_apd50s))
        possible_apd50s_per_apd90[apd90] = requested_apd50s

        if len(d_apd90s) > 1:  # Ensure AP table is equispaced in APD50
            raise Exception(f"AP table is not equispaced in APD50! Possible steps in APD90 in AP table are {d_apd50s}.")

    apd50_step = d_apd50s[0]

    times_ap = np.round(ap_table_2d[tuple(apd_tuples[0])][0], 8)  # Take any AP table times as they all share the time axis
    d_times_ap = np.unique(np.round(np.diff(times_ap), 8))  # Round to prevent float error

    if len(d_times_ap) > 1:  # Ensure AP table time axis is equispaced
        raise Exception(f"AP table time axis is not equispaced! Possible time steps are {d_times_ap}.")

    ap_time_res_s = d_times_ap[0]

    if not np.all(np.isin(times_sim_s, times_ap)):  # Ensure simulation times have a corresponding time in the AP table
        print(f"{times_sim_s=}")
        print(f"{times_ap=}")
        raise Exception("In the AP table time axis, there is not an entry corresponding to each simulation time point")

    n_times_ap = len(times_ap)
    n_apd90s = int((max_apd90 - min_apd90) / apd90_step) + 1
    n_apd50s = int((max_apd50 - min_apd50) / apd50_step) + 1

    # Every APD90 APD50 combo corresponding to a Vms array
    ap_table_arr = np.ones((n_apd90s, n_apd50s, n_times_ap)) * -1  # Sentinel value -1: no AP table entry
    ap_table_rmps = np.ones((n_apd90s, n_apd50s)) * -1  # Sentinel value -1: no AP table entry

    # Set up the final AP table array and AP RMP table array
    for apd_tuple in apd_tuples:
        apd90, apd50 = apd_tuple
        idx_apd90 = int((apd90 - min_apd90) / apd90_step)  # Zeroth column APD90 index
        idx_apd50 = int((apd50 - min_apd50) / apd50_step)  # First column APD50 index

        ap_table_arr[idx_apd90, idx_apd50] = ap_table_2d[tuple(apd_tuple)][1]
        ap_table_rmps[idx_apd90, idx_apd50] = ap_table_2d[tuple(apd_tuple)][1][-1]  # Last Vm value is taken as RMP

    ap_table_arr = ap_table_arr.astype(np.float32)
    ap_table_rmps = ap_table_rmps.astype(np.float32)

    return ap_table_arr, ap_table_rmps, min_apd90, max_apd90, min_apd50, max_apd50, apd90_step, apd50_step, ap_time_res_s, possible_apd50s_per_apd90


def setup_run_dir(main_dir, inferences_folder, benchmark_id, run_id, patient_id, mother_data_folder,
                  varying_angle, dx, angle_rot_deg, dataset_name, misc_suffix="", bench_dx=None):
    run_dir = f"{main_dir}/{inferences_folder}/{benchmark_id}/{run_id}"
    print(f"{run_dir=}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    if not os.path.exists(f"{run_dir}/pop_ids_and_diffs"):
        os.makedirs(f"{run_dir}/pop_ids_and_diffs")

    # Optionally copies to run dir some best QRS params, activation times, target QRS, and target QRS+Twave
    if mother_data_folder is not None:
        mother_dir = f"{main_dir}/{inferences_folder}/{benchmark_id}/{mother_data_folder}"

        if dataset_name == "oxdataset":
            shutil.copy(f"{mother_dir}/{benchmark_id}_bestqrsparams{misc_suffix}.npy",
                        f"{run_dir}/{benchmark_id}_bestqrsparams.npy")
            shutil.copy(f"{mother_dir}/{patient_id}_{dx}_activation_times{misc_suffix}.alg",
                        f"{run_dir}/{patient_id}_{dx}_activation_times.alg")
        elif dataset_name == "simulated_truths":
            shutil.copy(f"{mother_dir}/{patient_id}_500_ctrl_bestqrsparams{misc_suffix}.npy",
                        f"{run_dir}/{benchmark_id}_bestqrsparams.npy")
            shutil.copy(f"{mother_dir}/{patient_id}_{dx}_activation_times{misc_suffix}.alg",
                        f"{run_dir}/{patient_id}_{dx}_activation_times.alg")

        # Old code used for copying angle_rot-specific activation times and qrsparams
        """if not varying_angle:  # Then can use old activation times and params before we described angle rot
            shutil.copy(f"{mother_dir}/{patient_id}_{bench_dx}_ctrl_bestqrsparams_{angle_rot_deg}.npy",
                        f"{run_dir}/{patient_id}_{bench_dx}_ctrl_bestqrsparams.npy")
            shutil.copy(f"{mother_dir}/{patient_id}_{dx}_activation_times_{angle_rot_deg}.alg",
                        f"{run_dir}/{patient_id}_{dx}_activation_times.alg")
        else:  # Use activation times and params where we described the angle rot
            shutil.copy(f"{mother_dir}/{patient_id}_{bench_dx}_ctrl_bestqrsparams_{angle_rot_deg}.npy",
                        f"{run_dir}/{patient_id}_{bench_dx}_ctrl_bestqrsparams.npy")
            shutil.copy(f"{mother_dir}/{patient_id}_{dx}_activation_times_{angle_rot_deg}.alg",
                        f"{run_dir}/{patient_id}_{dx}_activation_times.alg")"""

        # For simulated targets for now
        if dataset_name == "simulated_truths":
            shutil.copy(f"{mother_dir}/leads_selected_qrs.npy", f"{run_dir}/leads_selected_qrs.npy")
            shutil.copy(f"{mother_dir}/leads_selected_qrsandtwave.npy", f"{run_dir}/leads_selected_qrsandtwave.npy")
        elif dataset_name == "oxdataset":
            # For clinical targets for now
            shutil.copy(f"{mother_dir}/{patient_id}_leads_subset1.npy", f"{run_dir}/leads_selected_qrs.npy")
            shutil.copy(f"{mother_dir}/{patient_id}_leads_subset2.npy", f"{run_dir}/leads_selected_qrsandtwave.npy")

    return run_dir


def create_sparse_adjacency_distance(adjacency_list):
    """ Create a sparse adjacency matrix of distances between cells from an adjacency list.

    Args:
        adjacency_list (dict): Dictionary where keys are cell indices (int), and values are lists of
            tuples (neighbor_index, displacement_vector). The displacement_vector is typically a NumPy array
            representing the spatial offset from the current cell to the neighbor in micrometers (um).

    Returns:
        csr_matrix: Sparse CSR matrix of shape (n_cells, n_cells), where each nonzero entry
            corresponds to the distance (in centimeters) between a cell and its neighbor.
    """
    row_indices, col_indices, data = [], [], []
    um_to_cm = 1e-4
    n_cells = len(adjacency_list)

    for idx, neighbors in adjacency_list.items():
        for neighbour_idx, displacement in neighbors:

            distance_um = np.linalg.norm(displacement)
            distance_cm = distance_um * um_to_cm
            row_indices.append(idx)
            col_indices.append(neighbour_idx)
            data.append(distance_cm)
    return csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_cells))