import log_analysis_functions as laf2

main_dir = "C:/Users/jammanadmin/Documents/Monoscription"

inferences_folder = "Inferences_qrs_oxdataset_local"
target, run_id = "DTI024", "run_20_0_0.0_calc_discrepancy_separate_scaling_0"
patient_id, dataset_name, coarse_dx = "DTI024", "oxdataset", 2000
repol, save_analysis = False, True
stop_thresh = 0.0005

#select_activation = ""  # reg tuners
select_activation = "_run_1024_0.0_0.0_calc_discrepancy_separate_scaling_0" #"_runtime_512_0.0_12_0.0_leads_sep_limb_prec_scale"

laf2.analyse_inf_log(main_dir, inferences_folder, dataset_name, patient_id, target, run_id, stop_thresh,
                select_activation, coarse_dx, repol, save_analysis)







