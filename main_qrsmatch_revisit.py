import sys

running_on_arc = True

if running_on_arc:
    scripts_dir = "/home/scat8499/monoscription_python/JAC_Py_Scripts"
    sys.path.append(scripts_dir)

import alg_utils2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import cache2
from constants2 import *
import log_inference2 as log2
import qrs_matching2 as qrsm2
import ecg2
import math
import utils2
import concurrent.futures
import random


def main():
    runtime_start = time.time()
    arg_names = ["benchmark_id", "n_processors", "n_tries", "inferences_folder", "angle_rot_deg", "dataset_name",
                 "axis_name", "elec_rad_translation_um", "elec_idxs_to_translate", "discrepancy_name",
                 "stop_thresh", "random_seed"]

    if running_on_arc:  # Setup arguments as in ARC run
        args = utils2.parse_args(arg_names)
        main_dir = "/data/coml-cardinal/scat8499/Monoscription"

        benchmark_id = args.benchmark_id
        n_processors = int(args.n_processors)
        n_tries = int(args.n_tries)
        inferences_folder = args.inferences_folder
        angle_rot_deg = float(args.angle_rot_deg)
        discrepancy_name = args.discrepancy_name
        dataset_name = args.dataset_name
        stop_thresh = float(args.stop_thresh)
        random_seed = int(args.random_seed)

        if dataset_name == "oxdataset":
            patient_id = benchmark_id  # oxdataset without dx, mesh_type (i.e. for DTI4309_1)
        elif dataset_name == "simulated_truths":
            patient_id = benchmark_id.split("_")[0]

        save_best_every_x = 200
        axis_name = args.axis_name
        elec_rad_translation_um = float(args.elec_rad_translation_um)
        s_clean = args.elec_idxs_to_translate.strip("[]")
        elec_idxs_to_translate = [int(x) for x in s_clean.split(",") if x.strip()]

    else:  # Setup arguments as in local run
        import addcopyfighandler
        main_dir = "C:/Users/jammanadmin/Documents/Monoscription"
        dataset_name = "simulated_truths"
        patient_id, bench_dx = "DTI003", 500
        inferences_folder = "Inferences_qrs_validation_local"
        stop_thresh = 0.00002
        random_seed = 0

        bench_type = "ctrl"
        n_tries, n_processors, save_best_every_x = 128, 3, 1
        angle_rot_deg, axis_name = 0, "lv_rv_vec_proj"
        elec_rad_translation_um, elec_idxs_to_translate = 0.0, []
        discrepancy_name = "calc_discrepancy_separate_scaling"  # calc_discrepancy_separate_scaling for oxdataset

        if dataset_name == "simulated_truths":
            benchmark_id = f"{patient_id}_{bench_dx}_{bench_type}" # monodomain ground truthsa
        elif dataset_name == "oxdataset":
            benchmark_id  = patient_id  # oxdataset

    ############################################# Key Parameters #######################################################
    run_id = f"run_{n_tries}_{angle_rot_deg}_{elec_rad_translation_um}_{discrepancy_name}_{random_seed}"
    dx, mesh_type = 2000, ""
    n_iterations, percent_cutoff = 1500, 87.5  # % accepted per iteration
    iter_dt_s, qrs_safety_s = 0.002, 0.02
    plot, use_fibers, use_best_guess, return_activation_times = 0, 0, 0, 1
    min_n_root_nodes, max_n_root_nodes, root_nodes_dist_apart_um = 6, 10, 5000  # root nodes
    v_endo_min, v_endo_max, v_endo_diff = 70, 190, 10  # possible v_endo range (cm/s)
    v_myo_min, v_myo_max, v_myo_diff = 20, 60, 10  # possible v_myo range (cm/s)

    # TODO TODO remove
    v_endo_min, v_endo_max, v_endo_diff = 70, 80, 10  # possible v_endo range (cm/s)
    v_myo_min, v_myo_max, v_myo_diff = 20, 30, 10  # possible v_myo range (cm/s)
    # todo todo
    log_every_x_iterations = 1
    window_size = 50
    ############################################# Best params ##########################################################
    params_best_guess = (85, 40), (911, 1092, 1652, 1660, 11504, 13627, 16022, 17508, 18137)
    #params_best_guess = np.load(f"{main_dir}/{inferences_folder}/{benchmark_id}_bestqrsparams.npy", allow_pickle=True)
    params_best_guess = tuple(params_best_guess)
    ####################################################################################################################

    random.seed(random_seed)
    np.random.seed(random_seed)

    if dataset_name == "simulated_truths":  # Before oxdataset
        # Loading alg mesh and cached geometrical information
        mesh_alg_name = f"{patient_id}_{dx}{mesh_type}.alg"
        mesh_alg_path = f"{main_dir}/Meshes_{dx}/{mesh_alg_name}"
        alg = alg_utils2.read_alg_mesh(mesh_alg_path)
        xs, ys, zs, lxs, lys, lzs = alg_utils2.unpack_alg_geometry(alg)
        dx = alg_utils2.get_dx(xs)
        n_cells = len(xs)
        cache_path = f"{main_dir}/Cache/{patient_id}_{dx}_cache.npy"
        mesh_info_dict = np.load(cache_path, allow_pickle=True).item()

        # Read from cache: endo surface, plane and electrode positions
        keys_to_read = ["endo_labels", "labels_meaning", "electrodes_xyz"]
        endo_labels, labels_meaning, electrodes_xyz = cache2.check_cache(mesh_info_dict, keys_to_read)
        lv_val, rv_val = labels_meaning["lv"], labels_meaning["rv"]
        lv_endo_idxs, rv_endo_idxs = np.where(endo_labels[:n_cells] == lv_val)[0], np.where(endo_labels[:n_cells] == rv_val)[0]
        endo_mask = np.zeros(n_cells)
        endo_mask[lv_endo_idxs], endo_mask[rv_endo_idxs] = 1, 1
        endo_idxs = np.where(endo_mask == 1)[0]
        xs_endo, ys_endo, zs_endo = xs[endo_idxs], ys[endo_idxs], zs[endo_idxs]
        alg_endo = alg_utils2.alg_from_xs(xs_endo, ys_endo, zs_endo)
        target_clinical = False
        leads_targ_name = "leads_selected_qrs"

    elif dataset_name == "oxdataset":  # After oxdataset
        mesh_alg_name = f"{patient_id}_{dx}_fields.alg"
        mesh_path = f"{main_dir}/Cache_Oxdataset/out/{mesh_alg_name}"
        alg = alg_utils2.read_alg_mesh(mesh_path)
        xs, ys, zs, *_ = alg_utils2.unpack_alg_geometry(alg)
        n_cells = len(xs)
        surface_info_field = alg[6]
        lv_endo_idxs, rv_endo_idxs = np.where(surface_info_field == 0)[0], np.where(surface_info_field == 1)[0]
        endo_mask = np.zeros(n_cells)
        endo_mask[lv_endo_idxs], endo_mask[rv_endo_idxs] = 1, 1
        endo_idxs = np.where(endo_mask == 1)[0]
        xs_endo, ys_endo, zs_endo = xs[endo_idxs], ys[endo_idxs], zs[endo_idxs]
        alg_endo = alg_utils2.alg_from_xs(xs_endo, ys_endo, zs_endo)
        patient_final_mesh_infos = np.load(f"{main_dir}/CardiacPersonalizationStudyVtks/alg_outs_final_trimmed/patient_final_mesh_infos_w_electrodes.npy", allow_pickle=True).item()
        electrodes_xyz = patient_final_mesh_infos[patient_id][-1]
        cache_path = f"{main_dir}/Cache_Oxdataset/{patient_id}_{dx}_cache.npy"
        mesh_info_dict = np.load(cache_path, allow_pickle=True).item()
        target_clinical = True
        leads_targ_name = f"{patient_id}_leads_subset1"

    # Setup run directory
    run_dir = f"{main_dir}/{inferences_folder}/{benchmark_id}/{run_id}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    np.save(f"{run_dir}/running.npy", np.array([1]))
    mother_dir = f"{main_dir}/{inferences_folder}/{benchmark_id}/mother_data"
    fast_download_folder = f"fast_{benchmark_id}"
    if not os.path.exists(f"{run_dir}/pop_ids_and_diffs"):
        os.makedirs(f"{run_dir}/pop_ids_and_diffs")

    v_endos = list(range(v_endo_min, v_endo_max + 1, v_endo_diff))
    v_myos = list(range(v_myo_min, v_myo_max + 1, v_myo_diff))

    if n_iterations < 100 and running_on_arc:
        raise Exception(f"{n_iterations=}")

    if (len(v_endos) == 1 or len(v_myos) == 1) and running_on_arc:
        print("Using debug setup still")
        print(f"{v_endos=}, {v_myos=}")
        raise Exception("ARC run with small v_endo, v_myo param space?")

    discrepancy_metrics = {"calc_discrepancy": qrsm2.calc_discrepancy,
                           "calc_discrepancy_separate_scaling": qrsm2.calc_discrepancy_separate_scaling}
    discrep_func = discrepancy_metrics[discrepancy_name]

    lead_names = LEAD_NAMES_12

    log_inf_params = {"main_dir": main_dir, "run_id": run_id, "patient_id": patient_id, "dx": dx,
                      "mesh_type": mesh_type,
                      "n_tries": n_tries, "n_iterations": n_iterations, "percent_cutoff": percent_cutoff,
                      "iter_dt_s": iter_dt_s,
                      "use_fibers": use_fibers, "target_clinical": target_clinical,
                      "min_n_root_nodes": min_n_root_nodes,
                      "max_n_root_nodes": max_n_root_nodes, "root_nodes_dist_apart_um": root_nodes_dist_apart_um,
                      "v_endos": v_endos, "v_myos": v_myos, "n_processors": n_processors,
                      "log_every_x_iterations": log_every_x_iterations, "angle_rot_deg": angle_rot_deg,
                      "lead_names": lead_names}

    if angle_rot_deg == "None" or angle_rot_deg == 0 or angle_rot_deg is None:
        varying_angle = False
    else:
        varying_angle = True

    # Unpack and pad target leads, and use this to make times_s
    leads_targ_in = np.load(f"{mother_dir}/{leads_targ_name}.npy", allow_pickle=True).item()
    leads_target, times_target_s, times_s, total_time_s = qrsm2.prepare_target_leads(leads_targ_in, iter_dt_s, qrs_safety_s)

    # Adjustment of electrode positions
    center_of_mass = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])

    # TODO adapt for clinical axes / different axis method
    keys_to_read = ["basal_plane_axis", "lv_rv_vec_proj", "final_axis2"]
    axis0, axis1, axis2 = cache2.check_cache(mesh_info_dict, keys_to_read)
    axes_dict = {"basal_plane_axis": axis0, "lv_rv_vec_proj": axis1, "final_axis2": axis2}

    electrodes_xyz = alg_utils2.rotate_electrodes(electrodes_xyz, axis0, axis1, axis2, axes_dict[axis_name], run_dir,
                                                 angle_rot_deg, varying_angle, center_of_mass)
    electrodes_xyz = alg_utils2.translate_electrodes(electrodes_xyz, elec_rad_translation_um, elec_idxs_to_translate,
                                                    center_of_mass, run_dir)

    # Preprocessing for pseudo ECG computation
    grid_dict = alg_utils2.make_grid_dictionary(xs, ys, zs)
    neighbour_arrays, *_ = ecg2.get_neighbour_arrays(xs, ys, zs, dx, grid_dict)
    elec_grads = ecg2.precompute_elec_grads(xs, ys, zs, electrodes_xyz, dx, neighbour_arrays).astype(np.float32)
    grid_endo_dict = alg_utils2.make_grid_dictionary(xs_endo, ys_endo, zs_endo)

    root_nodes_neighbour_dist_um = 2 * root_nodes_dist_apart_um
    # Set up candidate root node parameter space
    candidate_root_points, candidate_root_neighbours = qrsm2.mesh_subset_with_dist_constraint(alg_endo,
                                                                                           root_nodes_dist_apart_um,
                                                                                           root_nodes_neighbour_dist_um)

    # candidate_root_node_points contains indices (corresponding to the original alg mesh) of possible root nodes
    candidate_root_node_indices = []
    flag = [0 for _ in range(len(xs_endo))]
    for point in candidate_root_points:
        flag[grid_endo_dict[point]] = 1
        candidate_root_node_indices.append(grid_dict[point])

    if use_best_guess:
        n_tries, n_iterations = 1, 1  # Only 1 trial needed for best guess
        # Check root indices you are using are within the candidate root node indices list
        for root_index in params_best_guess[1]:
            if root_index not in candidate_root_node_indices:
                raise Exception("Root index in best guess is not in candidate root node indices list, check\
                                root_nodes_dist_apart_um")
        candidate_root_node_indices = params_best_guess[1]  # Only do dijkstra on root nodes in use if using best guess
        v_endos, v_myos = [params_best_guess[0][0]], [params_best_guess[0][1]]  # Confine v_params to best guess params

    # Used for activation time calculations in dijkstra (need 26 to ensure isotropic propagation is possible)
    adjacency_list_26 = ecg2.compute_adjacency_displacement(xs, ys, zs, dx, grid_dict, NEIGHBOURS_26)  # Post-fibers version (displacement vectors for fib projections)

    # Batch preparation of all all time matrices
    v_endos.sort()  # Sort v_params (ascending) as inc/decreasing v mutations relies on this
    v_myos.sort()
    v_space = len(v_endos) * len(v_myos)
    print(v_space, "velocity space", flush=True)

    param_args = [(v_endo, v_myo, adjacency_list_26, endo_mask, use_fibers, candidate_root_node_indices)
                  for v_endo in v_endos for v_myo in v_myos]
    batch_size = int(math.ceil(v_space / n_processors))
    batched_param_args = list(qrsm2.batcher(param_args, batch_size))

    all_all_time_matrices = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processors) as executor:
        results = executor.map(qrsm2.compute_time_matrix_batch, batched_param_args)
        for batch_result in results:
            all_all_time_matrices.update(batch_result)

    print(f"Max QRS time simulated: {max(times_s) * 1000}ms", flush=True)
    print(len(candidate_root_points), "root node parameter space", flush=True)

    log2.log_init_qrs(run_dir, log_inf_params, candidate_root_points, candidate_root_node_indices, times_target_s,
                     leads_target, alg, times_s)

    # Initialise root nodes and conduction velocity parameters
    current_iter_params = qrsm2.init_roots_and_vels(n_tries, min_n_root_nodes, max_n_root_nodes,
                                                 candidate_root_node_indices, v_endos, v_myos, params_best_guess,
                                                 use_best_guess)

    mutated_params, all_ids_and_diff_scores = {}, {}
    runtimes, iter_median_scores = [], []

    # Main iterative refinement of activation loop
    for iter_no in range(n_iterations):
        n_tries = len(current_iter_params)
        n_per_batch = int(round(n_tries / n_processors))
        n_per_batch = 1 if n_per_batch == 0 else n_per_batch
        tries = np.arange(n_tries)
        print(f"======================================== Iter {iter_no}: {n_tries} / {n_per_batch} ========================================", flush=True)

        # Compute electrode signals for simulations required this iteration
        all_electrodes, activation_times_s = qrsm2.batch_qrs_runner(n_tries, n_per_batch, qrsm2.pseudo_ecg_qrs, times_s,
                                                                    qrsm2.ap_heaviside, electrodes_xyz, elec_grads, dx,
                                                                    total_time_s, neighbour_arrays,
                                                                    all_all_time_matrices, current_iter_params)

        population_ids_check, population_ids, ids_and_ecgs_ats_params = {}, {}, {}

        for i_try in tries:
            # Conversion of params simulated this iteration to ids and record that the param_id is in pop already
            param_id = qrsm2.hash_qrs_param(current_iter_params[i_try])
            population_ids[i_try] = param_id
            population_ids_check[param_id] = 1

        all_leads_sim, population_diff_scores = qrsm2.analyse_sim_qrs_leads(all_electrodes, leads_target, times_s,
                                                                            times_target_s, discrep_func)

        for i_try, params in current_iter_params.items():  # Store all params, scores & activation times
            v_params, root_indices = params
            root_indices = list(root_indices)
            root_indices.sort()

            store_activation_times = np.round(np.array(activation_times_s[i_try]) * 1000)  # s to ms
            store_activation_times_ms = np.array(store_activation_times, dtype=np.uint16)

            param_id = qrsm2.hash_qrs_param((v_params, tuple(root_indices)))
            all_ids_and_diff_scores[param_id] = [population_diff_scores[i_try], iter_no]
            ids_and_ecgs_ats_params[param_id] = [all_leads_sim[i_try], store_activation_times_ms, params]

        # Retrieval of diff scores in the population but not simulated this iteration
        population_params = current_iter_params.copy()
        new_key = max(population_diff_scores.keys()) + 1

        for key, params in mutated_params.items():  # mutated_params is of the population size
            params = params[0], tuple(params[1])
            param_id = qrsm2.hash_qrs_param(params)

            # current_iter_params is of size n_tries of this iteration (unseen params)
            if param_id in all_ids_and_diff_scores and param_id not in population_ids_check:
                # Retrieving seen difference scores kept in the population but that were not simulated this iteration
                population_diff_scores[new_key] = all_ids_and_diff_scores[param_id][0]
                population_params[new_key] = params
                population_ids[new_key] = param_id
                population_ids_check[param_id] = 1
                new_key += 1

        scores = list(population_diff_scores.values())
        iter_median_scores.append(np.median(scores))

        keys_below, keys_above = ecg2.tie_aware_proportional_split(population_diff_scores, percent_cutoff)

        mutated_params = qrsm2.mutate_pop_params(keys_above, keys_below, population_params, alg, grid_dict,
                                                 candidate_root_node_indices, candidate_root_neighbours, v_endos, v_myos,
                                                 all_ids_and_diff_scores)

        next_iter_params = {}
        # Check which new root indices + velocity params have been analysed before already + set up next iteration
        next_iter_tries_ct = 0
        for key, params in mutated_params.items():
            v_params, root_indices = params
            param_id = qrsm2.hash_qrs_param((v_params, tuple(root_indices)))

            if param_id not in all_ids_and_diff_scores:  # Only simulate unseen params
                next_iter_params[next_iter_tries_ct] = params
                next_iter_tries_ct += 1

        # Update which root indices and v params to analyse next
        current_iter_params = next_iter_params

        min_diff_score = min(population_diff_scores.values())
        min_i_try = min(population_diff_scores, key=population_diff_scores.get)
        min_key = population_params[min_i_try]
        best_params = population_params[min_i_try]
        hash_best_param = qrsm2.hash_qrs_param(best_params)

        if hash_best_param in ids_and_ecgs_ats_params:
            # Then the new best reg params were found this iteration (so update all best values)
            best_ats = ids_and_ecgs_ats_params[hash_best_param][1]
            best_leads = ids_and_ecgs_ats_params[hash_best_param][0]

        if iter_no % save_best_every_x == 0:
            alg = alg[:6]
            alg.append(best_ats)
            alg_utils2.save_alg_mesh(f"{run_dir}/{fast_download_folder}/bestguess_best_params_{iter_no}.alg", alg)

        runtimes.append(time.time() - runtime_start)

        log2.log_progress_qrs(run_dir, iter_no, log_every_x_iterations, runtimes, all_ids_and_diff_scores,
                                       population_ids, population_diff_scores, ids_and_ecgs_ats_params,
                                       population_params)

        print(f"Best Params: {min_key}", flush=True)
        print(f"Unique Params Tested: {len(all_ids_and_diff_scores)}", flush=True)
        print(f"Min Score: {min_diff_score}", flush=True)

        converged, i_iter_final = ecg2.runtime_stop_condn(iter_no, iter_median_scores, window_size, stop_thresh)
        if converged:
            break
        # End of iteration loop

    if plot:
        alpha = qrsm2.find_optimal_scaling(best_leads, leads_target)
        best_leads_rescaled = {name: best_leads[name] * alpha for name in lead_names}
        ecg2.plot_ecg([times_s, times_target_s], [best_leads_rescaled, leads_target],
                     colors=["red", "black"], xlims=[0, 0.45])


if __name__ == '__main__':
    main()


