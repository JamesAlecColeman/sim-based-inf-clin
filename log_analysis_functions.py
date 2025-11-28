import utils
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from collections import defaultdict
import compare_distributions as comp2
import math


def get_max_i_iter(benchmark_run_dir, prefix = "all_params_and_diff_scores"):
    """Get the maximum completed iteration index from inference run

    Args:
        benchmark_run_dir (str): Path to inference run dir
        prefix (str, optional): Filename prefix to match files containing iteration data.
                                Default is "all_params_and_diff_scores"

    Returns:
        i_iter_maximum (int): Maximum iteration index found based on filenames
    """
    filenames = [f for f in os.listdir(benchmark_run_dir) if f.startswith(prefix)]
    #print(f"{benchmark_run_dir=}")
    #print(f"{filenames=}")
    #print(f"{prefix=}")

    # In case we find 2 all_params_and_diff_scores (run ended mid-save)
    possible_iters = []
    for f in filenames:
        possible_iters.append(int(f.split("_")[-1][:-4]))

    i_iter_maximum = min(possible_iters)
    return i_iter_maximum


def get_iteration_nos(benchmark_run_dir, i_population_name="population_params_and_diff_scores"):
    """Extract iteration numbers from inference run

    Args:
        benchmark_run_dir (str): Path to inference run dir
        i_population_name (str, optional): Prefix of the filenames to search for iteration nos
                                           Default is "population_params_and_diff_scores".

    Returns:
        iterations (list of int): Sorted list of iteration indices found in the filenames.
        n_iterations (int): Maximum iteration number found.
        log_every_x_iterations (int): Estimated logging frequency between saved iterations.
    """
    iterations = []
    i_population_filenames = utils.find_files(benchmark_run_dir, i_population_name)
    for filename in i_population_filenames:
        iterations.append(int(filename.split("_")[-1][:-4]))
    iterations.sort()

    if len(iterations) > 1:
        log_every_x_iterations = iterations[1] - iterations[0]
    else:
        log_every_x_iterations = 1

    n_iterations = max(iterations)
    return iterations, n_iterations, log_every_x_iterations


def get_scores(benchmark_run_dir, i_iter, i_population_name="population_params_and_diff_scores", repol=True):
    """Load diff and reg scores alongside params for a given iteration

    Args:
        benchmark_run_dir (str): Path to inference run dir
        i_iter (int): Iteration number to load.
        i_population_name (str, optional): Filename prefix for the population file.
                                           Default is "population_params_and_diff_scores".
        repol (bool, optional): Whether to use regularised scores (True) or raw diff scores (False).
                                Default is True.

    Returns:
        min_diff_score (float): Minimum unregularised (diff) score in the population.
        median_diff_score (float): Median unregularised (diff) score in the population.
        best_params_diff (array-like): Parameters corresponding to the minimum diff score.
        min_reg_score (float): Minimum regularised score (or diff score if `repol` is False).
        median_reg_score (float): Median regularised score (or diff score if `repol` is False).
        best_params_reg (array-like): Parameters corresponding to the minimum regularised score.
    """
    i_population_params_and_diff_scores = np.load(f"{benchmark_run_dir}/{i_population_name}_{i_iter}.npy",
                                                  allow_pickle=True)
    pop_params, pop_diff_scores = i_population_params_and_diff_scores[0], i_population_params_and_diff_scores[1]
    pop_reg_scores = i_population_params_and_diff_scores[2]

    if not repol:
        pop_reg_scores = pop_diff_scores

    i_tries = sorted(pop_params)#list(pop_params.keys())  # Keys in pop_params and pop_diff_scores dicts
    #i_tries.sort()

    pop_params_list, pop_diff_scores_list, pop_reg_scores_list = [], [], []

    for i_try in i_tries:
        pop_params_list.append(pop_params[i_try])
        pop_diff_scores_list.append(pop_diff_scores[i_try])
        pop_reg_scores_list.append(pop_reg_scores[i_try])

    """pop_params_list = [pop_params[i_try] for i_try in i_tries]
    pop_diff_scores_list = [pop_diff_scores[i_try] for i_try in i_tries]
    pop_reg_scores_list = [pop_reg_scores[i_try] for i_try in i_tries]"""


    min_diff_score = np.min(pop_diff_scores_list)
    i_min_diff_score = np.argmin(pop_diff_scores_list)
    median_diff_score = np.median(pop_diff_scores_list)
    best_params_diff = pop_params_list[i_min_diff_score]

    min_reg_score = np.min(pop_reg_scores_list)
    i_min_reg_score = np.argmin(pop_reg_scores_list)
    median_reg_score = np.median(pop_reg_scores_list)
    best_params_reg = pop_params_list[i_min_reg_score]

    return min_diff_score, median_diff_score, best_params_diff, min_reg_score, median_reg_score, best_params_reg


def apply_stop_condition(benchmark_run_dir, iterations, window_size=50, twave_diff_threshold=0.00002,
                         force_iter_final=None, plot=False,
                         i_population_name="population_params_and_diff_scores", repol=True):
    """Determine whether inference has converged based on stopping condition

    Applies a moving average over median regularised scores and checks whether
    the change has fallen below a specified threshold.

    Args:
        benchmark_run_dir (str): Path to inference run
        iterations (list of int): List of iteration numbers to evaluate.
        window_size (int, optional): Window size for moving average calculation. Default is 50.
        twave_diff_threshold (float, optional): Threshold for the moving average of score differences
                                                (absolute value). Default is -0.0001.
        force_iter_final (int or str, optional): Force use of a specific final iteration.
                                                 Use "max" to select the maximum available iteration.
        plot (bool, optional): Whether to display a plot of the moving average. Default is False.
        i_population_name (str, optional): Filename prefix of population score files.
                                           Default is "population_params_and_diff_scores".
        repol (bool, optional): Whether to use regularised scores (True) or raw diff scores (False).
                                Default is True.

    Returns:
        i_iter_final (int): Chosen final iteration based on stop condition or override.
        min_diff_score (float): Minimum unregularised score at final iteration.
        median_diff_score (float): Median unregularised score at final iteration.
        best_params_reg (array-like): Parameters corresponding to the best regularised score.
        min_reg_score (float): Minimum regularised score at final iteration.
        median_reg_score (float): Median regularised score at final iteration.
        abs_moving_avg (ndarray or None): Absolute moving average of score differences, or None if not computed.
    """
    if force_iter_final is None:
        min_scores, median_scores = [], []

        for i_iter in iterations:
            min_diff_score, median_diff_score, best_params_diff, min_reg_score, median_reg_score, best_params_reg = get_scores(benchmark_run_dir, i_iter, i_population_name=i_population_name, repol=repol)
            # Using regularised scores
            min_scores.append(min_reg_score)
            median_scores.append(median_reg_score)

        # Moving average over window size
        moving_avg = np.convolve(np.diff(median_scores), np.ones(window_size) / window_size, mode='same')
        abs_moving_avg = np.abs(moving_avg)
        abs_thresh = np.abs(twave_diff_threshold)
        below_threshold_indices = np.where(abs_moving_avg < abs_thresh)[0]

        if plot:
            plt.figure(1)
            plt.plot(abs_moving_avg, color="gray")
            plt.axhline(abs_thresh, color="red")
            plt.show()

        #print(f"{min(np.abs(moving_avg))=}")
        print(f"Minimum absolute moving average {min(np.abs(moving_avg)):.15f}")

        #print(moving_avg)

        if len(below_threshold_indices) == 0:
            raise Exception("Has not converged based on this threshold")


        i_iter_final = min(below_threshold_indices)
    else:
        i_iter_final = force_iter_final
        abs_moving_avg = None

        if force_iter_final == "max":
            i_iter_final = np.max(iterations)

    min_diff_score, median_diff_score, best_params_diff, min_reg_score, median_reg_score, best_params_reg = get_scores(benchmark_run_dir, i_iter_final, i_population_name=i_population_name, repol=repol)

    return int(i_iter_final), min_diff_score, median_diff_score, best_params_reg, min_reg_score, median_reg_score, abs_moving_avg


def ids_to_storage_iter_nos(pop_ids, all_ids_and_diff_scores):
    """Map population IDs to the iteration numbers where their data is stored.

    This identifies which iterations need to be loaded in order to retrieve ECGs and AT/RTs
    corresponding to the given population IDs.

    Args:
        pop_ids (list): List of population IDs whose data is needed.
        all_ids_and_diff_scores (dict): Dictionary mapping ID to a tuple where the second element
                                        is the iteration number it was stored in.

    Returns:
        iter_nos_to_pop_ids (defaultdict): Dictionary mapping iteration numbers to lists of population IDs
                                           that were stored in that iteration.
    """

    iter_nos_to_pop_ids = defaultdict(list)  # dict {iter_no: [params1, params2, ...]}
    for id in pop_ids:

        if id not in all_ids_and_diff_scores:
            print("Check there isn't a duplicate all_ids_and_diff_scores or download failure")

        iter_where_stored = all_ids_and_diff_scores[id][1]
        iter_nos_to_pop_ids[iter_where_stored].append(id)
    return iter_nos_to_pop_ids


def get_best_x_rts_or_ats(run_dir, iter_no, x_best_indices, all_ids_and_diff_scores, repol=True):
    """Retrieve RTs (or ATs) and ECG leads for the top X best-scoring parameter sets.

    Loads saved population data and identifies the top X individuals (based on regularised
    or unregularised scores). Then retrieves their corresponding RTs/ATs and ECG leads
    from storage.

    Args:
        run_dir (str): Path to the run directory containing population and ID mapping files.
        iter_no (int): Iteration number to load population scores from.
        x_best_indices (int): Number of best-scoring individuals to retrieve.
        all_ids_and_diff_scores (dict): Mapping of individual IDs to their storage iteration number
                                        and additional score/parameter data.
        repol (bool, optional): Whether to use regularised scores (True) or unregularised diff scores (False).
                                Default is True.

    Returns:
        best_x_rts (list): RT or AT values for the top X individuals (in order of score).
        best_x_reg_scores (ndarray): Regularised (or unregularised) scores of the top X individuals.
        best_x_leads (list): ECG lead dictionaries for the top X individuals.
    """
    pop_ids_and_diff_scores = np.load(f"{run_dir}/pop_ids_and_diffs/population_ids_and_diff_scores_{iter_no}.npy",
                                      allow_pickle=True)

    if repol:  # Use regularised scores
        pop_reg_scores = pop_ids_and_diff_scores[2]
        pop_diff_scores = pop_ids_and_diff_scores[1]
    else:  # Use diffs
        pop_reg_scores = pop_ids_and_diff_scores[1]
        pop_diff_scores = pop_ids_and_diff_scores[1]

    pop_ids = pop_ids_and_diff_scores[0]
    # pop_ids and pop_reg_scores are dicts like {i_try: id} so convert to arrays
    pop_ids, pop_reg_scores = np.array(list(pop_ids.values())), np.array(list(pop_reg_scores.values()))
    pop_diff_scores = np.array(list(pop_diff_scores.values()))

    # Get indices of the x best scores
    best_x_indices = np.argsort(pop_reg_scores)[:x_best_indices]
    best_x_ids = pop_ids[best_x_indices]
    best_x_reg_scores = pop_reg_scores[best_x_indices]
    best_x_diff_scores = pop_diff_scores[best_x_indices]

    # Finding iter nos of where RTs and ECGs saved of best x ids
    iter_nos_to_pop_ids = ids_to_storage_iter_nos(best_x_ids, all_ids_and_diff_scores)

    # Load RTs of the best x ids from where they were saved before
    best_x_ids_to_rts, best_x_ids_to_params, best_x_ids_to_leads = {}, {}, {}

    for iter_no2, ids in iter_nos_to_pop_ids.items():
        ids_and_rts_and_ecgs_temp = np.load(f"{run_dir}/ids_and_rts_and_ecgs_{iter_no2}.npy",
                                            allow_pickle=True).item()

        for id in ids:
            best_x_ids_to_leads[id] = ids_and_rts_and_ecgs_temp[id][0]
            best_x_ids_to_rts[id] = ids_and_rts_and_ecgs_temp[id][1]
            best_x_ids_to_params[id] = ids_and_rts_and_ecgs_temp[id][2]

    best_x_rts = [best_x_ids_to_rts[id] for id in best_x_ids]
    best_x_leads = [best_x_ids_to_leads[id] for id in best_x_ids]
    best_x_params = [best_x_ids_to_params[id] for id in best_x_ids]

    return best_x_rts, best_x_reg_scores, best_x_leads, best_x_params, best_x_diff_scores


def find_inference_runs(inferences_path):
    """Find inference targets and their runs in a given directory.

    Scans the specified inferences directory to identify target folders and the
    inference runs within each target, excluding specific folders like 'analysis'
    and 'mother_data'.

    Args:
        inferences_path (str): Path to the directory containing inference targets.

    Returns:
        targets_in_inf_folder (list of str): List of target folder names found in the directory,
                                             excluding 'analysis'.
        runs_in_targets (defaultdict(list)): Dictionary mapping each target to a list of
                                             its inference run folder names, excluding 'mother_data'.
    """
    targets_in_inf_folder = list(os.listdir(inferences_path))

    if "analysis" in targets_in_inf_folder:
        targets_in_inf_folder.remove("analysis")#

    if "alg_solutions.alg" in targets_in_inf_folder:
        targets_in_inf_folder.remove("alg_solutions.alg")

    # Detects inference runs for each target e.g. "DTI003_500_ctrl/runtime_512_-10.0"
    runs_in_targets = defaultdict(list)
    for target in targets_in_inf_folder:
        target_folder_path = f"{inferences_path}/{target}"
        target_folder_dir = list(os.listdir(target_folder_path))

        if "mother_data" in target_folder_dir:
            target_folder_dir.remove("mother_data")

        if len(target_folder_dir):
            runs_in_targets[target] = target_folder_dir

    return targets_in_inf_folder, runs_in_targets


def get_convergence_info(i_iter_final, repol, compare_to_truth, run_path, all_ids_and_diff_scores, x_best, iter_step,
                         i_iter_start, activation_ms, truth_times_ms, truth_apd90s_ms):
    # Find iteration-wise scores & comparisons to truths to plot convergence
    iters, iter_scores, iter_median_corrs, iter_median_absdiffs = [], [], [], []
    iter_median_corrs_apds, iter_median_absdiffs_apds = [], []
    iter_best_x_params = []

    corrs, mean_absdiffs, corrs_apds, mean_absdiffs_apds = None, None, None, None

    for iter_no in range(i_iter_start, i_iter_final, iter_step):
        best_x_times, best_x_reg_scores, best_x_leads, best_x_params, best_x_diffs = get_best_x_rts_or_ats(run_path, iter_no,
                                                                                                  x_best,
                                                                                                  all_ids_and_diff_scores,
                                                                                                  repol=repol)
        iter_best_x_params.append(best_x_params)

        if compare_to_truth:
            # Comparisons with ground truth for the best x solutions this iteration
            corrs = [comp2.correlation(times, truth_times_ms) for times in best_x_times]
            mean_absdiffs = [comp2.abs_diffs(times, truth_times_ms)[1] for times in best_x_times]

        if repol:  # Calculate and compare APDs
            best_x_apds = [repol_times - activation_ms for repol_times in best_x_times]
            if compare_to_truth:
                corrs_apds = [comp2.correlation(apds, truth_apd90s_ms) for apds in best_x_apds]
                mean_absdiffs_apds = [comp2.abs_diffs(apds, truth_apd90s_ms)[1] for apds in best_x_apds]

        # Store iteration-wise scores, correlations and absolute differences
        iters.append(iter_no)
        iter_scores.append(np.median(best_x_reg_scores))

        if compare_to_truth:
            iter_median_corrs.append(np.median(corrs))
            iter_median_absdiffs.append(np.median(mean_absdiffs))

            if repol:
                iter_median_corrs_apds.append(np.median(corrs_apds))
                iter_median_absdiffs_apds.append(np.median(mean_absdiffs_apds))
    return (iters, iter_scores, iter_median_corrs, iter_median_absdiffs, iter_median_corrs_apds,
            iter_median_absdiffs_apds, iter_best_x_params)


def get_ground_truth(benchmark_alg_path, repol):
    import alg_utils
    benchmark_alg = alg_utils.read_alg_mesh(benchmark_alg_path)  # APD90s, activation times, repolarisation times
    truth_apd90s_ms, truth_activations_s, truth_repols_ms = benchmark_alg[6], benchmark_alg[7], benchmark_alg[8]
    truth_activations_ms = truth_activations_s * 1000
    truth_times_ms = truth_activations_ms if not repol else truth_repols_ms
    return truth_times_ms, truth_apd90s_ms


def analyse_inf_log(main_dir, inferences_folder, dataset_name, patient_id, target, run_id, stop_thresh,
                    select_activation, coarse_dx, repol, save_analysis):
    import alg_utils
    inferences_path = f"{main_dir}/{inferences_folder}"
    if save_analysis:
        analysis_dir = f"{main_dir}/{inferences_folder}/analysis"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)

    mother_data_path = f"{inferences_path}/{target}/mother_data"
    mother_data_dir = list(os.listdir(mother_data_path))

    activation_times_count = 0  # check how many activation times are in mother data for this target
    for filename in mother_data_dir:
        if "activation_times" in filename and filename[-4:] == ".alg":
            activation_times_count += 1
    print(f"{activation_times_count} activation files in mother data")

    run_path = f"{inferences_path}/{target}/{run_id}"
    print("=======================================================================================")
    print(f"{target}/{run_id=}")

    i_iter_maximum = get_max_i_iter(run_path, prefix="all_ids_and_diff_scores")
    all_ids_and_diff_scores = np.load(f"{run_path}/all_ids_and_diff_scores_{i_iter_maximum}.npy", allow_pickle=True).item()
    iterations, n_iterations, log_every_x_iterations = get_iteration_nos(run_path, i_population_name="ids_and_rts_and_ecgs")

    # Application of stopping condition
    (i_iter_final, min_diff_score, median_diff_score, best_params_reg, min_reg_score, median_reg_score,
     abs_moving_avg) = apply_stop_condition(run_path, iterations, twave_diff_threshold=stop_thresh,
                                                i_population_name="pop_ids_and_diffs/population_ids_and_diff_scores",
                                                repol=repol, plot=0)
    print(f"Stopped at iteration {i_iter_final} of {n_iterations} iterations")
    print(f"min diff, reg scores = {float(min_diff_score)}, {float(min_reg_score)}")

    final_pop = np.load(f"{run_path}/pop_ids_and_diffs/population_ids_and_diff_scores_{i_iter_final}.npy",
                        allow_pickle=True)

    final_pop_diffs = list(final_pop[1].values())
    final_pop_regs = list(final_pop[2].values())

    log_inf_params = np.load(f"{run_path}/log_inf_params.npy", allow_pickle=True).item()
    n_tries = log_inf_params["n_tries"]
    percent_cutoff = log_inf_params["percent_cutoff"]
    n_solutions_to_cluster = int(math.floor(n_tries * (percent_cutoff / 100)))  # Remove latest mutation fraction

    best_x_times_final_iter, best_x_reg_scores_final_iter, best_x_leads_final_iter, best_x_params_final_iter, best_x_diff_scores_final_iter = get_best_x_rts_or_ats(
        run_path, i_iter_final, n_solutions_to_cluster, all_ids_and_diff_scores,
        repol=repol)

    representatives, labels, mean_reg_scores, n_clusters = comp2.solution_clusters(best_x_times_final_iter,
                                                                                   best_x_reg_scores_final_iter)
    cluster_id_lowest_score = min(mean_reg_scores, key=mean_reg_scores.get)
    print(f"{n_clusters=}")

    final_soln_idx_from_best_x_times_final_iter = None
    for cluster_id, medoid_soln_idx in representatives.items():
        soln = best_x_times_final_iter[medoid_soln_idx]

        if cluster_id == cluster_id_lowest_score:
            final_soln_idx_from_best_x_times_final_iter = medoid_soln_idx
            final_times_ms = soln  # Takes medoid of best scoring cluster at final iteration

    final_params = best_x_params_final_iter[final_soln_idx_from_best_x_times_final_iter]
    leads_sim_best = best_x_leads_final_iter[final_soln_idx_from_best_x_times_final_iter]

    activation_ms = None
    if repol:  # Load activation times from mother dir
        #select_activation = run_id.split("_")[-1]  # When using angle
        alg_activation = alg_utils.read_alg_mesh(f"{mother_data_path}/{patient_id}_{coarse_dx}_activation_times{select_activation}.alg")
        activation_s = alg_activation[-1]
        activation_ms = activation_s * 1000
        print(f"Run {run_id} using {patient_id}_{coarse_dx}_activation_times{select_activation}.alg as activation used")

    # Prepare to plot final ECG match to target
    times_s, times_target_s = np.load(f"{run_path}/times_s.npy"), np.load(f"{run_path}/times_target_s.npy")
    leads_target = np.load(f"{run_path}/leads_target.npy", allow_pickle=True).item()

    if repol:  #  QRS amplitudes need calculating specifically from the QRS subset of the target leads
        leads_selected_qrs = np.load(f"{run_path}/leads_selected_qrs.npy", allow_pickle=True).item()

    if repol:
        final_apd90s_ms = final_times_ms - activation_ms

    if save_analysis:
        # Local analysis dir
        np.save(f"{analysis_dir}/leads_sim_best_{patient_id}_{run_id}_{repol}.npy", leads_sim_best)
        np.save(f"{analysis_dir}/times_s_{patient_id}_{run_id}_{repol}.npy", times_s)
        np.save(f"{analysis_dir}/times_target_s_{patient_id}_{run_id}_{repol}.npy", times_target_s)
        np.save(f"{analysis_dir}/leads_target_{patient_id}_{run_id}_{repol}.npy", leads_target)
        np.save(f"{analysis_dir}/FINDIFF_{patient_id}_{run_id}_{repol}.npy", final_pop_diffs)

        if repol:
            np.save(f"{analysis_dir}/FINREG_{patient_id}_{run_id}.npy", final_pop_regs)

        # Oxdataset alg
        if dataset_name == "oxdataset":
            mesh_alg_name = f"{patient_id}_{coarse_dx}_fields.alg"
            mesh_path = f"{main_dir}/Cache_oxdataset/out/{mesh_alg_name}"
            alg = alg_utils.read_alg_mesh(mesh_path)
            alg = alg[:6]
        elif dataset_name == "simulated_truths":
            alg = alg_utils.read_alg_mesh(f"{main_dir}/Meshes_{coarse_dx}/{patient_id}_{coarse_dx}.alg")
        else:
            raise Exception(f"{dataset_name=}: oxdataset or simulated_truths?")

        if repol:
            np.save(f"{analysis_dir}/{patient_id}_{run_id}_leads_selected_qrs.npy", leads_selected_qrs)
            alg.append(final_times_ms)
            alg.append(final_apd90s_ms)
            alg_utils.save_alg_mesh(f"{analysis_dir}/{target}_repol_times_{run_id}.alg", alg)

            alg = alg[:6]
            alg.append(activation_ms)
            alg_utils.save_alg_mesh(f"{analysis_dir}/{patient_id}_{coarse_dx}_actvn_used_{run_id}.alg", alg)

            np.save(f"{analysis_dir}/{target}_besttwaveparams_{run_id}.npy", np.array(final_params, dtype=object))

        else:  # s conversion for activation
            alg.append(final_times_ms / 1000)
            alg_utils.save_alg_mesh(f"{analysis_dir}/{patient_id}_{coarse_dx}_activation_times_{run_id}.alg", alg)
            np.save(f"{analysis_dir}/{target}_bestqrsparams_{run_id}.npy", np.array(final_params, dtype=object))
