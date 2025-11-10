import qrs_matching2 as qrsm2
import twave_matching2 as twm2
import utils2
import alg_utils2
import numpy as np
import cache2
import ecg2
from constants2 import *
from smoothing2 import preprocessing_gaussian_smoothing_fourier
from scipy.sparse.csgraph import dijkstra
import addcopyfighandler

main_dir = "C:/Users/jammanadmin/Documents/Monoscription"
dataset_name = "oxdataset"
patient_id, dx = "DTI024", 2000

glob_0 = "global_analysis_qrs_oxdataset_final_1024"  # Activation global folder
glob_1 = "global_analysis_twave_oxdataset_final_actual_lambda200"  # Repolarisation global folder

misc_0 = "_run_1024_0.0_0.0_calc_discrepancy_separate_scaling_0"  # Activation misc suffix
misc_1 = "_reg_200.0_512_0.0_0.0_extended_floored_apexb_stopcondn_0"  # Repolarisation misc suffix

ap_table_name = "ap_table_2d_extended"
min_possible_apd90_ms, max_possible_apd90_ms = 150, 450

total_time_s = 0.600
iter_dt_qrs_s, iter_dt_twave_s = 0.005, 0.005

# Load fundamentals
if dataset_name == "simulated_truths":
    # Load alg old
    seg_name = "rvseg"
    mesh_dir = f"{main_dir}/Meshes_{dx}"
    mesh_filename = utils2.find_lvrv_thresh_used(mesh_dir, patient_id, dx, seg_name)
    alg = alg_utils2.read_alg_mesh(f"{mesh_dir}/{mesh_filename}")
    trans = alg[10]  # 0: endo, 1: epi
    apexb = alg[14]
    # Read from cache (old)
    cache_path = f"{main_dir}/Cache/{patient_id}_{dx}_cache.npy"
    mesh_info_dict = np.load(cache_path, allow_pickle=True).item()
    keys_to_read = ["electrodes_xyz"]
    (electrodes_xyz,) = cache2.check_cache(mesh_info_dict, keys_to_read)
# Load alg oxdataset
elif dataset_name == "oxdataset":
    mesh_alg_name = f"{patient_id}_{dx}_fields.alg"
    mesh_path = f"{main_dir}/Cache_Oxdataset/out/{mesh_alg_name}"
    alg = alg_utils2.read_alg_mesh(mesh_path)
    trans = alg[10]
    apexb = alg[12]
    # Read electrode posns (new)
    patient_final_mesh_infos = np.load(
        f"{main_dir}/CardiacPersonalizationStudyVtks/alg_outs_final_trimmed/patient_final_mesh_infos_w_electrodes.npy",
        allow_pickle=True).item()
    electrodes_xyz = patient_final_mesh_infos[patient_id][-1]
    cache_path = f"{main_dir}/Cache_Oxdataset/{patient_id}_{dx}_cache.npy"
    mesh_info_dict = np.load(cache_path, allow_pickle=True).item()

xs, ys, zs, *_ = alg_utils2.unpack_alg_geometry(alg)

# Preprocessing for pseudo ECG computation
grid_dict = alg_utils2.make_grid_dictionary(xs, ys, zs)
neighbour_arrays, neighbour_arrays2 = ecg2.get_neighbour_arrays(xs, ys, zs, dx, grid_dict)
elec_grads = ecg2.precompute_elec_grads(xs, ys, zs, electrodes_xyz, dx, neighbour_arrays).astype(np.float32)
neighbour_args = twm2.neighbour_arrays_to_args(neighbour_arrays, neighbour_arrays2)
adjacency_list_26 = ecg2.compute_adjacency_displacement(xs, ys, zs, dx, grid_dict, NEIGHBOURS_26)
adjacency_matrix = twm2.create_sparse_adjacency_distance(adjacency_list_26)

# Retrieve activation model used
activation_times_s = alg_utils2.read_alg_mesh(f"{main_dir}/{glob_0}/{patient_id}/0/{patient_id}_{dx}_activation_times{misc_0}.alg")[-1]
bestqrsparams = np.load(f"{main_dir}/{glob_0}/{patient_id}/0/{patient_id}_bestqrsparams{misc_0}.npy", allow_pickle=True)
v_params = bestqrsparams[0]  # just needs velocities for pseudo ecg checks

# Retrieve repolarisation model used
besttwaveparams = np.load(f"{main_dir}/{glob_1}/{patient_id}/1/{patient_id}_besttwaveparams{misc_1}.npy", allow_pickle=True).item()

# Prepare time axis for activation simulation
total_time_qrs_s = max(activation_times_s) + iter_dt_qrs_s
times_qrs_s = np.round(np.arange(0, total_time_qrs_s + iter_dt_qrs_s, iter_dt_qrs_s), decimals=6)
times_qrs_s = times_qrs_s[times_qrs_s <= total_time_qrs_s]  # Prevent overstepping beyond total_time_s

# Prepare time axis for repolarisation simulation
times_twave_s = np.round(np.arange(total_time_qrs_s + iter_dt_qrs_s, total_time_s + iter_dt_twave_s, iter_dt_twave_s), decimals=6)
times_s = np.concatenate((times_qrs_s, times_twave_s))

# Load AP table
ap_table_2d = np.load(f"{main_dir}/{ap_table_name}.npy", allow_pickle=True).item()  # APD: [times_map_s, vms_new, mKr, mK1]
ap_table_args = twm2.preprocess_2d_ap_table(ap_table_2d, times_s, 5)

# Set smoothing level
v_myo_cm_per_s = v_params[1]
conductivity = twm2.monoalg_cv_to_conductivity(v_myo_cm_per_s)
sigma_um = twm2.monoalg_conductivity_to_smoothing_sigma(conductivity)

# Precompute repolarisation requirements
x_i, y_i, z_i, vms_grid, dx, smoothed_mask = preprocessing_gaussian_smoothing_fourier(xs, ys, zs, sigma_um)
repol_args_2daptable = (x_i.astype(np.int32), y_i.astype(np.int32), z_i.astype(np.int32), vms_grid.astype(np.int32), dx,
                        smoothed_mask.astype(np.float32), sigma_um, total_time_qrs_s)

# Only compute dijkstras from disk idxs actually in the t wave params this time
apd90_disks = besttwaveparams["apd90_disks"]
disk_idxs = [dict(disk)["idx"] for disk in apd90_disks]
all_dijk_dists_cm = dijkstra(adjacency_matrix, return_predecessors=False, indices=disk_idxs)
all_dijk_dists_cm = {idx: all_dijk_dists_cm[i] for i, idx in enumerate(disk_idxs)}

apd90_params = twm2.params_to_apd90s_field_apexb(besttwaveparams, all_dijk_dists_cm, trans, apexb, min_possible_apd90_ms,
                                                 max_possible_apd90_ms)

n_qrs_idxs = len(times_qrs_s)

# plot a single ecg
all_vms = twm2.make_vms_field_2daptable(times_s, activation_times_s, besttwaveparams, ap_table_args,
                                        repol_args_2daptable, apd90_params)
electrodes_vs, _ = twm2.pseudo_ecg(times_s, electrodes_xyz, elec_grads, dx, neighbour_args, all_vms)
leads = ecg2.ten_electrodes_to_twelve_leads(electrodes_vs)

# Scale by setting I amp to 1mV, V1 amp to 1mV (separate limb, precordial for clin. ECG plot)
I_amp, V1_amp = np.max(leads["I"][:n_qrs_idxs]) - np.min(leads["I"][:n_qrs_idxs]), np.max(leads["V1"][:n_qrs_idxs]) - np.min(leads["V1"][:n_qrs_idxs])
leads_plot = {name: leads[name] / I_amp for name in LEAD_NAMES_LIMB_6}
for name in LEAD_NAMES_PREC_6:
    leads_plot[name] = leads[name] / V1_amp

ecg2.plot_ecg_clinical_style_testing([times_s], [leads_plot])

# Save all_vms somewhere
print(f"{all_vms.shape=}")
print(f"{times_s=}")
np.save(f"{main_dir}/all_vms_test.npy", all_vms)

# Example parameter sweep
"""
all_leads_plot = []

ap_shapes = np.arange(0.0, 1.0 + 0.1, 0.1)

for ap_shape_param in ap_shapes:

    besttwaveparams["ap_shape_param"] = ap_shape_param

    all_vms = twm2.make_vms_field_2daptable(times_s, activation_times_s, besttwaveparams, ap_table_args,
                                            repol_args_2daptable, apd90_params)
    electrodes_vs, _ = twm2.pseudo_ecg(times_s, electrodes_xyz, elec_grads, dx, neighbour_args, all_vms)
    leads = ecg2.ten_electrodes_to_twelve_leads(electrodes_vs)

    # Scale by setting I amp to 1mV, V1 amp to 1mV (separate limb, precordial for clin. ECG plot)
    I_amp, V1_amp = np.max(leads["I"][:n_qrs_idxs]) - np.min(leads["I"][:n_qrs_idxs]), np.max(leads["V1"][:n_qrs_idxs]) - np.min(leads["V1"][:n_qrs_idxs])
    leads_plot = {name: leads[name] / I_amp for name in LEAD_NAMES_LIMB_6}
    for name in LEAD_NAMES_PREC_6:
        leads_plot[name] = leads[name] / V1_amp

    all_leads_plot.append(leads_plot)

import matplotlib.pyplot as plt
cmap = plt.cm.viridis  # or 'plasma', 'coolwarm', etc.
colors = [cmap(ap_shape_param) for ap_shape_param in ap_shapes]

ecg2.plot_ecg_clinical_style([times_s for _ in range(len(all_leads_plot))], all_leads_plot, colors=colors)
"""
