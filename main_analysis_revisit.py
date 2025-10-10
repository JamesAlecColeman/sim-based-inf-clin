import alg_utils2
import ecg2
import log_analysis_functions2 as laf2
import matplotlib.pyplot as plt
import addcopyfighandler
import os
import compare_distributions2 as comp2
from constants2 import *
import random
import qrs_matching2 as qrsm2

main_dir = "C:/Users/jammanadmin/Documents/Monoscription"
glob_folder = None

inferences_folder, repol, save_analysis = "Inferences_twave_validation", 1, 1
dataset_name = "simulated_truths"
oxdataset = False
patient_id_select, run_id_select = None, None
patient_id_skip = None
stop_thresh, force_iter_final = 0.00002, None #"max" #"max"  #"max" #"max"  #"max"# None
#stop_thresh = 0.00002 is standard

plot_ecgs = 1
cap_at_converged_iter = False  # Stops plots going all the way to 1000+ iterations

compare_to_truth, benchmarks_folder = True, "New_Benchmarks_APDs"
iter_step, x_best = 100, 51  # View approximate convergence every iter_step iterations, just the x_best solutions

i_iter_start = 0

#select_activation = ""  # reg tuners
select_activation = "_run_512_0.0_0.0_calc_discrepancy_separate_scaling" #"_runtime_512_0.0_12_0.0_leads_sep_limb_prec_scale"


coarse_dx = 2000

inferences_path = f"{main_dir}/{inferences_folder}"
targets_in_inf_folder, runs_in_targets = laf2.find_inference_runs(inferences_path)
lead_names = LEAD_NAMES_12

if save_analysis:
    analysis_dir = f"{main_dir}/{inferences_folder}/analysis"
    glob_analysis_dir = f"{main_dir}/{glob_folder}"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

# Prepare figure to plot iteration-wise scores and comparisons to ground truths
width_px, height_px, dpi = 850, 750, 100
width_in, height_in = width_px / dpi, height_px / dpi
fig, axs = plt.subplots(3, 2, figsize=(width_in, height_in), dpi=dpi)
n_rows, n_cols = axs.shape

all_run_scores, all_run_corrs, all_run_absdiffs = {}, {}, {}

all_t_corrs = []
all_apd_corrs = []

n_targ = len(targets_in_inf_folder)
for i_targ, target in enumerate(runs_in_targets.keys()):  # E.g. now in "Inferences_Folder/DTI003_500_ctrl"
    print(f"{target=}")
    #print("=")
    #print("=")
    if patient_id_select is not None:

        target_fields = target.split("_")

        if len(patient_id_select.split("_")) != 1:  # Handling patient_ids with 2 _s
            if target != patient_id_select:
                continue
        else:
            if target_fields[0] != patient_id_select:
                continue

    if patient_id_skip is not None:
        # Ensure skip list is iterable
        if isinstance(patient_id_skip, str):
            patient_id_skip = [patient_id_skip]

        # Extract patient ID from target and skip if it's in the list
        patient_id = target  # Pay attention to this; this isn't always true
        #patient_id = target.split("_")[0]
        if patient_id in patient_id_skip:
            continue

    target_fields = target.split("_")

    if len(target_fields) == 1:
        patient_id = target_fields[0]
    else:
        if len(target_fields) < 3:
            patient_id = target
        else:
            patient_id, fine_dx, mesh_type = target_fields[0], target_fields[1], target_fields[2]
            benchmark_id = f"{patient_id}_{fine_dx}_{mesh_type}"

    n_runs = len(runs_in_targets[target])

    mother_data_path = f"{inferences_path}/{target}/mother_data"
    mother_data_dir = list(os.listdir(mother_data_path))

    truth_times_ms, truth_apd90s_ms = None, None

    if compare_to_truth:
        # Load ground truth activation/repolarisation sequence for this target
        benchmark_alg_path = f"{main_dir}/{benchmarks_folder}/{patient_id}_{coarse_dx}_{mesh_type}_APDs.alg"
        benchmark_alg = alg_utils2.read_alg_mesh(benchmark_alg_path)  # APD90s, activation times, repolarisation times
        truth_apd90s_ms, truth_activations_s, truth_repols_ms = benchmark_alg[6], benchmark_alg[7], benchmark_alg[8]
        truth_activations_ms =  truth_activations_s * 1000

        activation_times_count = 0
        for filename in mother_data_dir:
            if "activation_times" in filename and filename[-4:] == ".alg":
                activation_times_count += 1

        print(f"{activation_times_count} activation files in mother data, {n_runs=}")

        truth_times_ms = truth_activations_ms if not repol else truth_repols_ms


    for i_run, run_id in enumerate(runs_in_targets[target]):  # E.g. now in ""DTI003_500_ctrl/runtime_512_-10.0""
        if run_id_select is not None:
            if run_id != run_id_select:
                continue

        run_path = f"{inferences_path}/{target}/{run_id}"
        print("=======================================================================================")
        print(f"{target}/{run_id=}")

        i_iter_maximum = laf2.get_max_i_iter(run_path, prefix="all_ids_and_diff_scores")
        all_ids_and_diff_scores = np.load(f"{run_path}/all_ids_and_diff_scores_{i_iter_maximum}.npy", allow_pickle=True).item()
        iterations, n_iterations, log_every_x_iterations = laf2.get_iteration_nos(run_path, i_population_name="ids_and_rts_and_ecgs")

        # Application of stopping condition
        (i_iter_final, min_diff_score, median_diff_score, best_params_reg, min_reg_score, median_reg_score,
         abs_moving_avg) = laf2.apply_stop_condition(run_path, iterations, twave_diff_threshold=stop_thresh,
                                                    i_population_name="pop_ids_and_diffs/population_ids_and_diff_scores",
                                                    repol=repol, force_iter_final=force_iter_final)

        print(f"Stopped at iteration {i_iter_final} of {n_iterations} iterations")
        print(f"min diff, reg scores = {float(min_diff_score)}, {float(min_reg_score)}")

        final_pop = np.load(f"{run_path}/pop_ids_and_diffs/population_ids_and_diff_scores_{i_iter_final}.npy",
                            allow_pickle=True)

        final_pop_diffs = list(final_pop[1].values())
        final_pop_regs = list(final_pop[2].values())


        # Extract best model at final iteration
        best_x_times, best_x_reg_scores, best_x_leads, best_x_params = laf2.get_best_x_rts_or_ats(run_path, i_iter_final, 1, all_ids_and_diff_scores,
                                                                              repol=repol)

        print(f"{best_x_params=}")
        activation_ms = None

        if repol:  # Load activation times from mother dir
            #select_activation = run_id.split("_")[-1]  # When using angle
            alg_activation = alg_utils2.read_alg_mesh(f"{mother_data_path}/{patient_id}_{coarse_dx}_activation_times{select_activation}.alg")
            activation_s = alg_activation[-1]
            activation_ms = activation_s * 1000
            print(f"Run {run_id} using {patient_id}_{coarse_dx}_activation_times{select_activation}.alg as activation used")


        # Find iteration-wise scores & comparisons to truths to plot convergence
        iters, iter_scores, iter_median_corrs, iter_median_absdiffs = [], [], [], []

        iter_median_corrs_apds, iter_median_absdiffs_apds = [], []

        iter_best_x_params = []

        corrs, mean_absdiffs, corrs_apds, mean_absdiffs_apds = None, None, None, None

        if cap_at_converged_iter:
            i_iter_maximum = i_iter_final

        for iter_no in range(i_iter_start, i_iter_maximum, iter_step):
            best_x_times, best_x_reg_scores, best_x_leads, best_x_params = laf2.get_best_x_rts_or_ats(run_path, iter_no, x_best, all_ids_and_diff_scores,
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

        if not repol:
            pass
            # Scatterplots of vendo vmyo
            """# Looking into whether v_endo, v_myo have some compensation issue
            print(f"{iters=}")
            print(f"{iter_best_x_params=}")
            iter_v_endos, iter_v_myos = [[] for _ in iters], [[] for _ in iters]

            for m, best_x_params in enumerate(iter_best_x_params):
                for params in best_x_params:
                    iter_v_endos[m].append(params[0][0])
                    iter_v_myos[m].append(params[0][1])

            x = np.array(iter_v_myos)
            y = np.array(iter_v_endos)
            num_m, num_points = x.shape
            x_flat = x.flatten()
            y_flat = y.flatten()
            colors_per_point = np.repeat(np.array(iters), num_points)  # shape (num_m * num_points,)
            plt.figure(24)
            sc = plt.scatter(x_flat, y_flat, c=colors_per_point, cmap='cool', alpha=1.0)
            cbar = plt.colorbar(sc)
            cbar.set_label('Iteration number')
            plt.title("(v_endo, v_myo) of best 10% solutions per iteration")
            plt.xlabel('v_myo (cm/s)', fontsize=18)
            plt.ylabel('v_endo (cm/s)', fontsize=18)
            plt.xlim([10, 100])
            plt.ylim([60, 200])
            plt.show()"""


        # Same color for the same run
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_colors = len(color_cycle)
        color_index = (i_run * n_targ + i_targ) % n_colors
        col = color_cycle[color_index]

        axs[0, 0].plot(iters, iter_scores, label=f"{target}_{run_id}", color=col)  # Scores
        axs[0, 0].axvline(x=i_iter_final, linestyle="--", color=col, linewidth=1.0)  # Stopping iteration

        if compare_to_truth:
            axs[1, 0].plot(iters, iter_median_corrs, label=run_id, color=col)  # Correlations
            axs[1, 1].plot(iters, iter_median_absdiffs, label=run_id, color=col)  # Absolute differences

            if repol:
                axs[2, 0].plot(iters, iter_median_corrs_apds, label=run_id, color=col)  # Correlations
                axs[2, 1].plot(iters, iter_median_absdiffs_apds, label=run_id, color=col)  # Absolute differences

        # Prepare to plot final ECG match to target
        times_s, times_target_s = np.load(f"{run_path}/times_s.npy"), np.load(f"{run_path}/times_target_s.npy")
        leads_target = np.load(f"{run_path}/leads_target.npy", allow_pickle=True).item()

        leads_sim_best = best_x_leads[0]

        if repol:
            # QRS amplitudes need calculating specifically from the QRS subset of the target leads
            leads_selected_qrs = np.load(f"{run_path}/leads_selected_qrs.npy", allow_pickle=True).item()
            target_qrs_times_s = leads_selected_qrs["I"][0]
            target_qrs_times_ms = target_qrs_times_s * 1000
            target_qrs_leads = {lead_name: leads_selected_qrs[lead_name][1] for lead_name in lead_names}
            target_leads_amps = {name: np.max(target_qrs_leads[name]) - np.min(target_qrs_leads[name]) for name in lead_names}
        else:
            # QRS amplitudes taken from general target leads because target only contains QRS
            target_leads_amps = {name: np.max(leads_target[name]) - np.min(leads_target[name]) for name in lead_names}
            target_qrs_leads = leads_target

        # Normalisation of QRSes for comparison
        target_leads_normed = {name: leads_target[name] / target_leads_amps[name] for name in lead_names}

        sim_leads_amps = {name: np.max(leads_sim_best[name]) - np.min(leads_sim_best[name]) for name in lead_names}
        sim_leads_normed = {name: leads_sim_best[name] / sim_leads_amps[name] for name in lead_names}

        # Discern QRS from T wave inside sim times_s by iter_dts; assumes iter_dt_activation < iter_dt_repol
        time_diffs = np.diff(times_s)

        possible_time_diffs = np.unique(np.round(time_diffs, decimals=5))
        if len(possible_time_diffs) > 2:
            #print(f"{times_s=}")
            #print(f"{time_diffs=}")
            #print(f"{possible_time_diffs=}")
            pass

        min_time_diff = np.min(possible_time_diffs)
        sim_qrs_idxs = np.where(np.isclose(time_diffs, min_time_diff, atol=1e-5))[0]

        if not repol:
            sim_qrs_idxs = np.unique(np.append(sim_qrs_idxs, np.max(sim_qrs_idxs) + 1))  # 1 idx offset correction because of np.diffs

            if len(sim_qrs_idxs) != len(leads_sim_best["I"]):
                raise Exception("Number of QRS idxs != length of leads, and we are QRS matching")

        times_qrs_s = times_s[sim_qrs_idxs]

        # Note leads_target has all its original time points saved, not necessarily matched to leads_sim_best
        # So re-match time points

        target_idxs = ecg2.match_sim_and_target_times(times_s, times_target_s)
        target_qrs_idxs = ecg2.match_sim_and_target_times(times_qrs_s, times_target_s)
        leads_target_justcompare = {name: leads_target[name][target_idxs] for name in lead_names}
        leads_target_normed = {name: target_leads_normed[name][target_idxs] for name in lead_names}

        ecg_fig_no = random.randint(1_000_000, 9_999_999)

        # Only needed for clin. ECG plotting
        units_2pt5uV_to_mV = 0.0025  # Conversion scale factor for clinical ECGs from units of 2.5uV to mV

        # PLOTTING OF LEAD-SPECIFIC NORMALISATION ECG
        """ecg.plot_ecg([times_s, times_target_s[target_idxs]],
                     [sim_leads_normed, leads_target_normed],
                     xlims=[0, 0.45], colors=["red", "black"], fig_no=ecg_fig_no, title=target+run_id, show=True,
                     labels=["Inferred", "Target"], axes_off=False, ylabel="Scaled Signal (unitless)")

        # PLOTTING OF ONE OPTIMAL GLOBAL SCALING FACTOR ECG (PRESERVES ALL INTER-LEAD RATIOS)
        sim_qrs_leads = {lead: leads_sim_best[lead][sim_qrs_idxs] for lead in leads_target}
        target_qrs_leads_masked = {lead: target_qrs_leads[lead][target_qrs_idxs] for lead in leads_target}
        s_opt = qrsm.find_optimal_scaling(sim_qrs_leads, target_qrs_leads_masked)
        fleads_sim_best = {lead: s_opt * leads_sim_best[lead] for lead in leads_sim_best}
        ecg.plot_ecg([times_s, times_target_s[target_idxs]],[fleads_sim_best, leads_target_justcompare],
            xlims=[0, 0.45], colors=["red", "black"], fig_no=ecg_fig_no, title=target + run_id, show=True,
            labels=["Inferred", "Target"], axes_off=False,ylabel="Signal (mV)", sharey=True,
                     rescale_signal=units_2pt5uV_to_mV)"""

        # PLOTTING OF TWO OPTIMAL SCALING FACTORS, ONE FOR PREC, ONE FOR LIMB
        limb_leads, prec_leads = ["I", "II", "III", "aVR", "aVL", "aVF"], ["V1", "V2", "V3", "V4", "V5", "V6"]

        target_qrs_amps = {name: np.max(target_qrs_leads[name]) - np.min(target_qrs_leads[name]) for name in
                           target_qrs_leads}

        #if not repol and not oxdataset:  # Rescale target to make "I" 1mV and "V1" 1mV
        if not oxdataset:  # Rescale target to make "I" 1mV and "V1" 1mV
            target_qrs_leads = {name: target_qrs_leads[name] / (target_qrs_amps["I"] * units_2pt5uV_to_mV) if name in limb_leads else
                                target_qrs_leads[name] for name in target_qrs_leads}
            target_qrs_leads = {
                name: target_qrs_leads[name] / (target_qrs_amps["V1"] * units_2pt5uV_to_mV) if name in prec_leads else
                target_qrs_leads[name] for name in target_qrs_leads}

        target_qrs_amp_ratios = {name: target_qrs_amps[name] / target_qrs_amps["I"] for name in
                           target_qrs_leads}

        leads_target_justcompare = {name: leads_target_justcompare[name] / (target_qrs_amps["I"] * units_2pt5uV_to_mV) if name in limb_leads else
                                leads_target_justcompare[name] / (target_qrs_amps["V1"] * units_2pt5uV_to_mV) for name in lead_names}



        #print(f"{target_qrs_amp_ratios=}")

        sim_limb_qrs = {lead: leads_sim_best[lead][sim_qrs_idxs] for lead in limb_leads if lead in leads_sim_best}
        target_limb_qrs = {lead: target_qrs_leads[lead][target_qrs_idxs] for lead in limb_leads if lead in target_qrs_leads}
        sim_prec_qrs = {lead: leads_sim_best[lead][sim_qrs_idxs] for lead in prec_leads if lead in leads_sim_best}
        target_prec_qrs = {lead: target_qrs_leads[lead][target_qrs_idxs] for lead in prec_leads if lead in target_qrs_leads}
        s_limb, s_prec = qrsm2.find_optimal_scaling(sim_limb_qrs, target_limb_qrs), qrsm2.find_optimal_scaling(sim_prec_qrs, target_prec_qrs)

        #print(f"{s_limb=}, {s_prec=}")

        fleads_sim_best_dual = {}
        for lead in leads_sim_best:
            if lead in limb_leads:
                scale = s_limb
            elif lead in prec_leads:
                scale = s_prec
            fleads_sim_best_dual[lead] = scale * leads_sim_best[lead]

        if not repol:
            plot_target = {name: target_qrs_leads[name][target_idxs] for name in target_qrs_leads}
        else:
            plot_target = leads_target_justcompare

        if plot_ecgs:
            ecg2.plot_ecg_clinical_style([times_s, times_target_s[target_idxs]],
                         [fleads_sim_best_dual, plot_target],
                         xlims=[0, 0.45], colors=["red", "black"], fig_no=ecg_fig_no + 1,
                         title=target + run_id + " (Dual Scaling)", show=False,
                         labels=["Simulated", "Target"], axes_off=True, ylabel="Signal (mV)",
                         sharey=True, rescale_signal=units_2pt5uV_to_mV,
                         linestyles=["-", "--"])

        # TODO revisit plotting of the ECG match to preserve inter-lead amplitude ratios

        # Finding iter nos of where times + ECGs are stored for best params
        iter_nos_to_pop_ids = laf2.ids_to_storage_iter_nos([best_params_reg], all_ids_and_diff_scores)

        # Load times of the best x ids from where they were saved before
        best_x_ids_to_rts, best_x_ids_to_params, best_x_ids_to_leads = {}, {}, {}
        for iter_no2, ids in iter_nos_to_pop_ids.items():
            ids_and_rts_and_ecgs_temp = np.load(f"{run_path}/ids_and_rts_and_ecgs_{iter_no2}.npy",
                                                allow_pickle=True).item()
            for id in ids:
                best_x_ids_to_params[id] = ids_and_rts_and_ecgs_temp[id][2]
        best_x_params = [best_x_ids_to_params[id] for id in [best_params_reg]]

        #print(f"{best_x_params=}")

        #print(f"{best_x_params=}")
        final_times_ms = best_x_times[0]

        if compare_to_truth:
            corr_final = comp2.correlation(final_times_ms, truth_times_ms)
            print(f"{run_id} {corr_final=}")
            mean_absdiffs_final = comp2.abs_diffs(final_times_ms, truth_times_ms)[1]
            all_run_corrs[f"{target}/{run_id}"] = corr_final
            all_run_absdiffs[f"{target}/{run_id}"] = mean_absdiffs_final
            all_t_corrs.append(corr_final)

            if repol:
                final_apds_ms = final_times_ms - activation_ms
                corr_apd_final = comp2.correlation(final_apds_ms, truth_apd90s_ms)
                all_apd_corrs.append(corr_apd_final)
                print(f"{round(corr_final, 3)=}")
                print(f"{round(corr_apd_final, 3)=}")

        all_run_scores[f"{target}/{run_id}"] = min_reg_score

        if save_analysis:
            if glob_folder is not None:
                glob_pt_dir = f"{glob_analysis_dir}/{patient_id}/{repol}"
                if not os.path.exists(glob_pt_dir):
                    os.makedirs(glob_pt_dir)


            # Local analysis dir
            np.save(f"{analysis_dir}/leads_sim_best_{patient_id}_{run_id}_{repol}.npy", leads_sim_best)
            np.save(f"{analysis_dir}/times_s_{patient_id}_{run_id}_{repol}.npy", times_s)
            np.save(f"{analysis_dir}/times_target_s_{patient_id}_{run_id}_{repol}.npy", times_target_s)
            np.save(f"{analysis_dir}/leads_target_{patient_id}_{run_id}_{repol}.npy", leads_target)
            np.save(f"{analysis_dir}/FINDIFF_{patient_id}_{run_id}_{repol}.npy", final_pop_diffs)

            if glob_folder is not None: # Global analysis dir
                np.save(f"{glob_pt_dir}/leads_sim_best_{patient_id}_{run_id}_{repol}.npy", leads_sim_best)
                np.save(f"{glob_pt_dir}/times_s_{patient_id}_{run_id}_{repol}.npy", times_s)
                np.save(f"{glob_pt_dir}/times_target_s_{patient_id}_{run_id}_{repol}.npy", times_target_s)
                np.save(f"{glob_pt_dir}/leads_target_{patient_id}_{run_id}_{repol}.npy", leads_target)
                np.save(f"{glob_pt_dir}/FINDIFF_{patient_id}_{run_id}_{repol}.npy", final_pop_diffs)


            if repol:
                np.save(f"{analysis_dir}/FINREG_{patient_id}_{run_id}.npy", final_pop_regs)

                if glob_folder is not None:
                    np.save(f"{glob_pt_dir}/FINREG_{patient_id}_{run_id}.npy", final_pop_regs)


            # Oxdataset alg
            if dataset_name == "oxdataset":
                mesh_alg_name = f"{patient_id}_{coarse_dx}_fields.alg"
                mesh_path = f"{main_dir}/Cache_oxdataset/out/{mesh_alg_name}"
                alg = alg_utils2.read_alg_mesh(mesh_path)
                alg = alg[:6]
            elif dataset_name == "simulated_truths":
                alg = alg_utils2.read_alg_mesh(f"{main_dir}/Meshes_{coarse_dx}/{patient_id}_{coarse_dx}.alg")

            if repol:
                np.save(f"{analysis_dir}/{patient_id}_{run_id}_leads_selected_qrs.npy", leads_selected_qrs)

                if glob_folder is not None:
                    np.save(f"{glob_pt_dir}/{patient_id}_{run_id}_leads_selected_qrs.npy", leads_selected_qrs)

                alg.append(final_times_ms)
                final_apd90s_ms = best_x_times[0] - activation_ms
                alg.append(final_apd90s_ms)
                #alg_utils.save_alg_mesh(f"{analysis_dir}/{patient_id}_{coarse_dx}_repol_times.alg", alg)
                alg_utils2.save_alg_mesh(f"{analysis_dir}/{benchmark_id}_repol_times_{run_id}.alg", alg)
                if glob_folder is not None:
                    alg_utils2.save_alg_mesh(f"{glob_pt_dir}/{benchmark_id}_repol_times_{run_id}.alg", alg)

                alg = alg[:6]
                alg.append(activation_ms)
                alg_utils2.save_alg_mesh(f"{analysis_dir}/{patient_id}_{coarse_dx}_actvn_used_{run_id}.alg", alg)
                if glob_folder is not None:
                    alg_utils2.save_alg_mesh(f"{glob_pt_dir}/{patient_id}_{coarse_dx}_actvn_used_{run_id}.alg", alg)

                np.save(f"{analysis_dir}/{target}_besttwaveparams_{run_id}.npy", np.array(best_x_params[0], dtype=object))
                if glob_folder is not None:
                    np.save(f"{glob_pt_dir}/{target}_besttwaveparams_{run_id}.npy", np.array(best_x_params[0], dtype=object))

            else:  # s conversion for activation
                alg.append(final_times_ms / 1000)
                alg_utils2.save_alg_mesh(f"{analysis_dir}/{patient_id}_{coarse_dx}_activation_times_{run_id}.alg", alg)
                if glob_folder is not None:
                    alg_utils2.save_alg_mesh(f"{glob_pt_dir}/{patient_id}_{coarse_dx}_activation_times_{run_id}.alg", alg)
                np.save(f"{analysis_dir}/{target}_bestqrsparams_{run_id}.npy", np.array(best_x_params[0], dtype=object))
                if glob_folder is not None:
                    np.save(f"{glob_pt_dir}/{target}_bestqrsparams_{run_id}.npy", np.array(best_x_params[0], dtype=object))

axs[0, 0].set_ylabel("Scores")
axs[0, 0].legend(fontsize='x-small', borderpad=0.1, labelspacing=0.2, handletextpad=0.2, loc='best')

if compare_to_truth:
    activn_repoln_string = "ATs" if not repol else "RTs"

    axs[1, 0].set_ylabel(f"{activn_repoln_string} Spearman r")
    axs[1, 0].set_ylim([0, 1])

    axs[1, 1].set_ylabel(f"{activn_repoln_string} Abs Diffs (ms)")

    if repol:
        axs[2, 0].set_ylabel("APD90s Spearman r")
        axs[2, 0].set_ylim([0, 1])
        axs[2, 1].set_ylabel("APD90s Abs Diffs (ms)")

# Plot style choices to apply to all subplots
for i in range(n_rows):
    for j in range(n_cols):
        axs[i, j].tick_params(axis='both', which='both', length=8, direction='inout', labelsize=12)
        axs[i, j].xaxis.label.set_fontsize(16)
        axs[i, j].yaxis.label.set_fontsize(16)
        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['bottom'].set_visible(True)
        axs[i, j].spines['left'].set_visible(True)
        axs[i, j].grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
        axs[i, j].set_xlabel("Iterations")

plt.tight_layout()
plt.show()

# At-a-glance summary of run final results as a bar plot
width_px, height_px, dpi = 850, 750, 100
width_in = width_px / dpi
height_in = height_px / dpi
fig, axs = plt.subplots(3, 2, figsize=(width_in, height_in), dpi=dpi)
n_rows, n_cols = axs.shape
axs = axs.flatten()

# Data sources and titles
data_sources = [
    ('Scores per Run', all_run_scores),
    ('Correlations per Run', all_run_corrs),
    ('Absolute Differences per Run', all_run_absdiffs)
]

# Plot bar charts in first 3 subplots
for i, (title, data) in enumerate(data_sources):
    labels = list(data.keys())
    values = list(data.values())
    axs[i].bar(labels, values, color='skyblue')
    axs[i].set_title(title)
    axs[i].set_xlabel('Run ID')
    axs[i].set_ylabel('Value')
    axs[i].tick_params(axis='x', rotation=20, labelsize=8)

for j in range(len(data_sources), len(axs)):
    fig.delaxes(axs[j])

print(f"{round(np.mean(all_t_corrs), 2)} {round(np.std(all_t_corrs), 2)}")
print(f"{round(np.mean(all_apd_corrs), 2)} {round(np.std(all_apd_corrs), 2)}")

all_t_corrs = [float(round(x, 2)) for x in all_t_corrs]
all_apd_corrs = [float(round(x, 2)) for x in all_apd_corrs]

print(f"{all_t_corrs=}")
print(f"{all_apd_corrs=}")

plt.tight_layout()
plt.show()