import sys

running_on_arc = True

if running_on_arc:
    scripts_dir = "/home/scat8499/monoscription_python/JAC_Py_Scripts"
    sys.path.append(scripts_dir)

import alg_utils2
import os
from constants2 import *
import cache2
from smoothing2 import preprocessing_gaussian_smoothing_fourier
import numpy as np
import time
import log_inference2 as log
from scipy.sparse.csgraph import dijkstra
import ecg2
import twave_matching2 as twm2
import utils2


def main():
    runtime_start = time.time()

    arg_names = ["benchmark_id", "lambda_reg", "seg_name", "n_processors", "n_tries", "inferences_folder",
                 "angle_rot_deg", "axis_name", "elec_rad_translation_um", "elec_idxs_to_translate",
                 "dataset_name", "stop_thresh"]

    if running_on_arc:  # ARC run setup
        args = utils2.parse_args(arg_names)

        main_dir = "/data/coml-cardinal/scat8499/Monoscription"

        benchmark_id = args.benchmark_id
        lambda_reg = float(args.lambda_reg)
        n_processors = int(args.n_processors)
        n_tries = int(args.n_tries)
        inferences_folder = args.inferences_folder
        dataset_name = args.dataset_name
        stop_thresh = float(args.stop_thresh)

        if args.angle_rot_deg != "None":
            angle_rot_deg = float(args.angle_rot_deg)
        else:
            angle_rot_deg = args.angle_rot_deg

        if dataset_name == "oxdataset":
            patient_id = benchmark_id  # oxdataset without dx, mesh_type (i.e. for DTI4309_1)
        elif dataset_name == "simulated_truths":
            patient_id = benchmark_id.split("_")[0]

        save_best_every_x = 100
        axis_name = args.axis_name
        elec_rad_translation_um = float(args.elec_rad_translation_um)
        s_clean = args.elec_idxs_to_translate.strip("[]")
        elec_idxs_to_translate = [int(x) for x in s_clean.split(",") if x.strip()]

    else:  # Local run setup
        import addcopyfighandler
        main_dir = "C:/Users/jammanadmin/Documents/Monoscription"
        dataset_name = "simulated_truths"
        patient_id, bench_dx = "DTI003", 500
        inferences_folder = "Inferences_twave_validation_local"
        stop_thresh = 0.00002

        bench_type = "hcmbig"
        n_tries, n_processors, save_best_every_x, lambda_reg = 32, 6, 1, 300.0
        angle_rot_deg, axis_name = 0, "lv_rv_vec_proj"
        elec_rad_translation_um, elec_idxs_to_translate = 0.0, []  # 4, 5, 6, 7, 8, 9 is V1-V6

        if dataset_name == "simulated_truths":
            benchmark_id = f"{patient_id}_{bench_dx}_{bench_type}"  # monodomain ground truthsa
        elif dataset_name == "oxdataset":
            benchmark_id = patient_id  # oxdataset

    ############################################# Key Parameters #######################################################
    run_id = f"reg_{lambda_reg}_{n_tries}_{angle_rot_deg}_{elec_rad_translation_um}_extended_floored_apexb_stopcondn"
    misc_suffix = f"_run_512_0.0_0.0_calc_discrepancy_separate_scaling"  # For loading activation times
    dx = 2000
    n_iterations, percent_cutoff = 1600, 87.5
    iter_dt_activation_s, iter_dt_repol_s, twave_safety_s = 0.002, 0.010, 0.050
    activation_start_s = 0.000
    plot, use_fibers, use_best_guess = 0, 0, 0
    no_seg_dir = f"{main_dir}/no_segments"
    log_every_x_iterations = 1
    window_size = 50
    ap_table_name = "ap_table_2d_extended"
    min_possible_apd90_ms, max_possible_apd90_ms, apd90_snapping_ms = 150, 450, 1
    mother_data_folder = "mother_data"
    fast_download_folder = f"fast_{benchmark_id}"
    ############################################# Best params ##########################################################
    if use_best_guess:
       best_params_preload = np.load(f"{main_dir}/global_analysis/{patient_id}/1/{patient_id}_besttwaveparams_reg_300.0_512_0.0_0.0_extended_floored_apexb_stopcondn.npy", allow_pickle=True).item()
    ####################################################################################################################

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

    if angle_rot_deg == "None" or angle_rot_deg == 0 or angle_rot_deg is None:
        varying_angle = False
    else:
        varying_angle = True

    # Setup run directory
    run_dir = twm2.setup_run_dir(main_dir, inferences_folder, benchmark_id, run_id, patient_id, mother_data_folder,
                                 varying_angle, dx, angle_rot_deg, dataset_name, misc_suffix=misc_suffix)

    alg = alg[:6]
    n_cells = len(alg[0])
    xs, ys, zs, *_ = alg_utils2.unpack_alg_geometry(alg)

    # Prepare dijkstra distances for radius-wise apd manipulation
    dijk_dist_path = f"{no_seg_dir}/{patient_id}_{dx}_dijk_dists.npy"
    print("Prepare dijkstra distances")
    if os.path.exists(dijk_dist_path):  # Load dijkstra dists if already saved
        all_dijk_dists_cm = np.load(dijk_dist_path)
    else:  # Prepare dijkstra neighbourhoods for all points
        grid_dict = alg_utils2.make_grid_dictionary(xs, ys, zs)
        adjacency_list_26 = ecg2.compute_adjacency_displacement(xs, ys, zs, dx, grid_dict,
                                                              NEIGHBOURS_26)  # Post-fibers version (displacement vectors for fib projections)
        adjacency_matrix = twm2.create_sparse_adjacency_distance(adjacency_list_26)
        all_dijk_dists_cm = dijkstra(adjacency_matrix, return_predecessors=False)  # Distances in cm
        if not running_on_arc:
            np.save(dijk_dist_path, all_dijk_dists_cm)

    all_dijk_dists_cm = all_dijk_dists_cm.astype(np.float16)
    lead_names = LEAD_NAMES_12

    # Get target leads into correct form
    leads_target_qrs = np.load(f"{run_dir}/leads_selected_qrs.npy", allow_pickle=True).item()
    leads_target_qrsandtwave = np.load(f"{run_dir}/leads_selected_qrsandtwave.npy", allow_pickle=True).item()

    leads_target, times_target_s, activation_cutoff_s = twm2.prepare_target_leads(leads_target_qrs, leads_target_qrsandtwave,
                                                                             twave_safety_s)
    # Activation times setup
    times_activation_s = np.round(
        np.arange(activation_start_s, activation_cutoff_s + iter_dt_activation_s, iter_dt_activation_s), decimals=6)
    times_activation_s = times_activation_s[times_activation_s <= activation_cutoff_s]  # Prevent overstepping

    # Repol times setup
    repol_start_s = activation_cutoff_s + iter_dt_activation_s
    total_time_s = max(times_target_s)
    times_repol_s = np.round(np.arange(repol_start_s, total_time_s + iter_dt_repol_s, iter_dt_repol_s), decimals=6)
    times_repol_s = times_repol_s[times_repol_s <= total_time_s]  # Prevent overstepping

    times_s = np.concatenate((times_activation_s, times_repol_s))
    print(f"Simulating up to {round(total_time_s * 1000)}ms")

    # Load AP table
    ap_table_2d = np.load(f"{main_dir}/{ap_table_name}.npy", allow_pickle=True).item()  # APD: [times_map_s, vms_new, mKr, mK1]
    ap_table_args = twm2.preprocess_2d_ap_table(ap_table_2d, times_s, 5)

    (ap_table_arr, ap_table_rmps, min_apd90, max_apd90, min_apd50, max_apd50,
     apd90_step, apd50_step, ap_time_res_s, possible_apd50s_per_apd90) = ap_table_args

    # Use of existing activation times from previous QRS personalisation
    qrsparams = np.load(f"{run_dir}/{benchmark_id}_bestqrsparams.npy", allow_pickle=True)
    v_myo_cm_per_s = qrsparams[0][1]
    conductivity = twm2.monoalg_cv_to_conductivity(v_myo_cm_per_s)
    sigma_um_param = twm2.monoalg_conductivity_to_smoothing_sigma(conductivity)

    mesh_alg_activation_name = f"{patient_id}_{dx}_activation_times.alg"
    mesh_alg_activation_path = f"{run_dir}/{mesh_alg_activation_name}"
    alg_activation = alg_utils2.read_alg_mesh(mesh_alg_activation_path)
    activation_times_s = np.array(alg_activation[-1])
    activation_times_s = np.round(activation_times_s / ap_time_res_s) * ap_time_res_s

    sigma_um = sigma_um_param

    # Prepare dict to log inference parameters
    log_inf_params = {"main_dir": main_dir, "run_id": run_id, "patient_id": patient_id, "dx": dx,
                      "n_tries": n_tries, "n_iterations": n_iterations, "percent_cutoff": percent_cutoff,
                      "iter_dt_activation_s": iter_dt_activation_s, "iter_dt_repol_s": iter_dt_repol_s,
                      "use_fibers": use_fibers, "min_possible_apd90_ms": min_possible_apd90_ms,
                      "max_possible_apd90_ms": max_possible_apd90_ms, "apd90_snapping_ms": apd90_snapping_ms,
                      "sigma_um_param": sigma_um_param, "n_processors": n_processors,
                      "log_every_x_iterations": log_every_x_iterations,
                      "qrsparams": qrsparams, "ap_table_name": ap_table_name, "window_size":window_size,
                      "stop_thresh": stop_thresh, "twave_safety_s": twave_safety_s}

    center_of_mass = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])

    keys_to_read = ["basal_plane_axis", "lv_rv_vec_proj", "final_axis2"]
    axis0, axis1, axis2 = cache2.check_cache(mesh_info_dict, keys_to_read)
    axes_dict = {"basal_plane_axis": axis0, "lv_rv_vec_proj": axis1, "final_axis2": axis2}

    electrodes_xyz = alg_utils2.rotate_electrodes(electrodes_xyz, axis0, axis1, axis2, axes_dict[axis_name], run_dir,
                                                  angle_rot_deg, varying_angle, center_of_mass)
    electrodes_xyz = alg_utils2.translate_electrodes(electrodes_xyz, elec_rad_translation_um, elec_idxs_to_translate,
                                                    center_of_mass, run_dir)

    # Preprocessing for pseudo ECG computation
    grid_dict = alg_utils2.make_grid_dictionary(xs, ys, zs)
    neighbour_arrays, neighbour_arrays2 = ecg2.get_neighbour_arrays(xs, ys, zs, dx, grid_dict)
    elec_grads = ecg2.precompute_elec_grads(xs, ys, zs, electrodes_xyz, dx, neighbour_arrays).astype(np.float32)

    # Preprocess parts of smoothing
    x_i, y_i, z_i, vms_grid, dx, smoothed_mask = preprocessing_gaussian_smoothing_fourier(xs, ys, zs, sigma_um)

    possible_apd90s_ms = np.arange(min_possible_apd90_ms, max_possible_apd90_ms + 1, apd90_snapping_ms, dtype=np.int16)

    if use_best_guess:
        n_tries, plot, n_iterations = 1, 1, 1
        current_iter_params = {0: best_params_preload}
    else:
        current_iter_params = twm2.init_twave_params_apexb(n_tries, possible_apd90s_ms, n_cells)

    #  Ensure current_iter_params only contains dicts
    for key, value in current_iter_params.items():
        if isinstance(value, frozenset):
            current_iter_params[key] = dict(value)  # Convert frozenset to dict

    if sigma_um <= 1000.0:
        print(f"Using low smoothing parameter {sigma_um=}, appropriate if setting based on ground truth APD field")

    # Pass in all the preprocessed arguments used for T wave computations
    repol_args_2daptable = (x_i.astype(np.int32), y_i.astype(np.int32), z_i.astype(np.int32), vms_grid.astype(np.int32), dx, smoothed_mask.astype(np.float32), sigma_um, activation_cutoff_s)
    neighbour_args = twm2.neighbour_arrays_to_args(neighbour_arrays, neighbour_arrays2)

    current_iter_apd90s = {}  # Convert current_iter_params to current_iter_apd90s
    for i_try, params in current_iter_params.items():
        current_iter_apd90s[i_try] = twm2.params_to_apd90s_field_apexb(params, all_dijk_dists_cm, trans, apexb,
                                                                       min_possible_apd90_ms, max_possible_apd90_ms)
    all_activation_times_s = [activation_times_s]
    # QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS QRS
    (all_electrodes, _, all_vms_activation,_) = \
        (twm2.batch_ecg_runner(1, 1, twm2.pseudo_ecg, times_activation_s, electrodes_xyz, elec_grads,
                               dx, neighbour_args, current_iter_apd90s, current_iter_params, all_activation_times_s,
                               repol_args_2daptable, ap_table_args, times_s, None, return_vms=True))

    leads_qrs_sim = ecg2.ten_electrodes_to_twelve_leads(all_electrodes[0])
    all_vms_activation = all_vms_activation[0]  # dict to array

    if not use_best_guess:
        log.log_init_twave(run_dir, log_inf_params, times_target_s, leads_target, alg, times_s, activation_times_s)

    mutated_params, all_ids_and_diff_scores, all_ids_and_grad_norms = {}, {}, {}
    runtimes, iter_median_reg_scores = [], []
    compute_repolarisation_times = True

    # Main iterative refinement of repolarisation loop
    for iter_no in range(n_iterations):
        n_tries = len(current_iter_params)
        n_per_batch = int(round(n_tries / n_processors))
        n_per_batch = 1 if n_per_batch == 0 else n_per_batch
        tries = np.arange(n_tries)
        print(f"======================================== Iter {iter_no}: {n_tries} / {n_per_batch} ========================================", flush=True)

        all_activation_times_s = [activation_times_s for _ in range(n_tries)]  # All use same activation sequence

        current_iter_apd90s = {}  # Convert current_iter_params to current_iter_apd90s
        for i_try, params in current_iter_params.items():
            current_iter_apd90s[i_try] = twm2.params_to_apd90s_field_apexb(params, all_dijk_dists_cm, trans, apexb,
                                                                           min_possible_apd90_ms, max_possible_apd90_ms)

        # T wave T wave T wave T wave T wave T wave T wave T wave T wave T wave T wave T wave T wave T wave T wave
        (all_electrodes, all_repol_times, _, all_mean_mean_grad_norms) = \
            (twm2.batch_ecg_runner(n_tries, n_per_batch, twm2.pseudo_ecg, times_repol_s, electrodes_xyz, elec_grads, dx,
                                   neighbour_args,  current_iter_apd90s, current_iter_params, all_activation_times_s,
                                   repol_args_2daptable, ap_table_args, times_s, all_vms_activation, return_vms=False))

        population_diff_scores, population_reg_scores, all_leads_sim, all_leads_full_ecg_sim_notrescaled = {}, {}, {}, {}
        population_ids, population_ids_check, ids_and_ecgs_rts_params, population_grad_norms = {}, {}, {}, {}

        for i_try in tries:
            leads_twave_sim = ecg2.ten_electrodes_to_twelve_leads(all_electrodes[i_try])
            # For saving of full unscaled ECG
            all_leads_full_ecg_sim_notrescaled[i_try] = {name: np.concatenate([np.array(leads_qrs_sim[name]), np.array(leads_twave_sim[name])]) for name in lead_names}
            population_diff_scores[i_try], times_target_subset_s, all_leads_sim[i_try], target_full_ecg_leads_normed = \
                (twm2.get_diff_score(times_repol_s, leads_twave_sim, times_target_s, leads_target, leads_qrs_sim,
                                     times_activation_s))
            population_grad_norms[i_try] = all_mean_mean_grad_norms[i_try]
            population_reg_scores[i_try] = population_diff_scores[i_try] + lambda_reg * all_mean_mean_grad_norms[i_try]
            param_id = twm2.hash_twave_param(current_iter_params[i_try])
            population_ids[i_try] = param_id
            population_ids_check[param_id] = 1

        for i_try, twave_param in current_iter_params.items():  # Store all params, diff scores, times and leads

            store_repol_times_ms = None

            if compute_repolarisation_times:
                store_repol_times_ms = np.round(np.array(all_repol_times[i_try]) * 1000)  # s to ms
                store_repol_times_ms = np.array(store_repol_times_ms, dtype=np.uint16)

            param_id = twm2.hash_twave_param(twave_param)
            all_ids_and_diff_scores[param_id] = [population_diff_scores[i_try], iter_no, population_reg_scores[i_try]]

            # Now saving non-rescaled leads
            ids_and_ecgs_rts_params[param_id] = [all_leads_full_ecg_sim_notrescaled[i_try], store_repol_times_ms, twave_param]
            all_ids_and_grad_norms[param_id] = all_mean_mean_grad_norms[i_try]

        # Retrieval of diff scores in the population but not simulated this iteration
        population_params = current_iter_params.copy()
        new_key = max(population_diff_scores.keys()) + 1

        for key, twave_param in mutated_params.items():  # mutated_params is of the population size
            param_id = twm2.hash_twave_param(twave_param)
            if param_id in all_ids_and_diff_scores and param_id not in population_ids_check:
                # Retrieving seen difference scores kept in the population but that were not simulated this iteration
                # Frozenset when used as key (to be hashable), dict when used as value (to be usable)
                population_diff_scores[new_key] = all_ids_and_diff_scores[param_id][0]
                population_reg_scores[new_key] = all_ids_and_diff_scores[param_id][2]
                population_grad_norms[new_key] = all_ids_and_grad_norms[param_id]
                population_params[new_key] = dict(twave_param)
                population_ids[new_key] = param_id
                population_ids_check[param_id] = 1
                new_key += 1

        regularised_scores = list(population_reg_scores.values())
        iter_median_reg_scores.append(np.median(regularised_scores))
        keys_below, keys_above = ecg2.tie_aware_proportional_split(population_reg_scores, percent_cutoff)

        if len(keys_above) == 0 and not use_best_guess:
            print(f"{regularised_scores=}")
            raise Exception(f"{keys_above=}, no worse keys found! Many models probably have the same reg score (are different models actually different?)")

        min_diff_score = min(population_diff_scores.values())
        min_reg_score = min(population_reg_scores.values())
        min_i_try_reg = min(population_reg_scores, key=population_reg_scores.get)
        best_reg_params = population_params[min_i_try_reg]
        hash_best_param = twm2.hash_twave_param(best_reg_params)

        mutated_params = twm2.mutate_twave_params_2daptable_apexb(keys_above, keys_below, population_params,
                                                          min_possible_apd90_ms, max_possible_apd90_ms,
                                                          all_dijk_dists_cm, trans, all_ids_and_diff_scores)

        next_iter_params = {}
        next_iter_tries_ct = 0
        for key, twave_param in mutated_params.items():
            param_id = twm2.hash_twave_param(twave_param)
            if param_id not in all_ids_and_diff_scores:  # Only simulate unseen params
                next_iter_params[next_iter_tries_ct] = twave_param
                next_iter_tries_ct += 1
        current_iter_params = next_iter_params

        print(f"Best Reg Params: {best_reg_params}", flush=True)
        print(f"Unique Params Tested: {len(all_ids_and_diff_scores)}", flush=True)
        print(f"Min Score: {min_diff_score} / {min_reg_score}", flush=True)

        if hash_best_param in ids_and_ecgs_rts_params:
            # Then the new best reg params were found this iteration (so update all best values)
            best_rts = ids_and_ecgs_rts_params[hash_best_param][1]
            best_leads = ids_and_ecgs_rts_params[hash_best_param][0]

        if iter_no % save_best_every_x == 0:  # Fast-download folder for at-a-glance inference evaluation
            apd90s_best_ms = twm2.params_to_apd90s_field_apexb(best_reg_params, all_dijk_dists_cm, trans, apexb,
                                                               min_possible_apd90_ms, max_possible_apd90_ms)
            alg = alg[:6]
            alg.append(apd90s_best_ms)  # APD params
            best_apds = best_rts - activation_times_s * 1000  # Post smoothing
            alg.append(best_apds)
            alg.append(best_rts)
            alg_utils2.save_alg_mesh(f"{run_dir}/{fast_download_folder}/best_reg_params_{iter_no}.alg", alg)
            np.save(f"{run_dir}/{fast_download_folder}/best_leads_{iter_no}.npy", best_leads)

        runtime_end = time.time()
        runtime_current_total = runtime_end - runtime_start
        runtimes.append(runtime_current_total)

        if not use_best_guess:
            log.log_progress_twave_perturb(run_dir, iter_no, log_every_x_iterations, runtimes, all_ids_and_diff_scores,
                                           population_ids, population_diff_scores, ids_and_ecgs_rts_params,
                                           population_reg_scores, population_params)

        converged, i_iter_final = ecg2.runtime_stop_condn(iter_no, iter_median_reg_scores, window_size, stop_thresh)
        if converged:
            break
        # End of iteration loop

    if plot:
        ecg2.plot_ecg([times_s, times_target_s],
                     [all_leads_sim[0], target_full_ecg_leads_normed],
                     colors=["red", "black"], labels=["Inferred", "Target"])

if __name__ == '__main__':
    main()
