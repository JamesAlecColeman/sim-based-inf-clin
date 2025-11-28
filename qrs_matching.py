import numpy as np
import ecg
import alg_utils
from constants import *
from itertools import islice
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import random
import concurrent.futures
import math
import hashlib


def mutate_activation_params(params, alg, grid_dict, candidate_root_node_indices, candidate_root_neighbours,
                             v_endos_cm_per_s, v_myos_cm_per_s, n_random_mutations=2, p_exploration=0.3, p_velocity=0.3):
    """ Mutates activation parameters of a single activation model

    Args:
        params (tuple tuple): activation model params (v_endo_param, v_myo_param), (root_idx1, ...)
        alg (list): alg mesh
        grid_dict (dict): coordinate to mesh idx {(x, y, z): idx, ...}
        candidate_root_node_indices (int list): idxs of allowed root node positions
        candidate_root_neighbours (dict): neighbours of allowed root node posns stored as {(x, y, z): [(x0, y0, z0), ...], ...}
        v_endos_cm_per_s (float list): possible endocardial conduction velocities
        v_myos_cm_per_s (float list): possible myocardial conduction velocities
        n_random_mutations (int): number of root node changes per mutation
        p_exploration (float): probability of explore-type mutation
        p_velocity (float): probability of mutating a conduction velocity param

   Returns:
       params (tuple tuple): mutated activation model params (v_endo_param, v_myo_param), (root_idx1, ...)
    """
    xs, ys, zs, *_ = alg_utils.unpack_alg_geometry(alg)
    v_params, root_indices = params
    new_v_params, new_root_indices = v_params, list(root_indices).copy()

    if random.random() < p_velocity:  # Mutations applied to endo and myo conduction velocities
        v_endo_cm_per_s, v_myo_cm_per_s = v_params
        v_to_mutate = random.choice([0, 1])  # Randomly choose endo or myo velocity to mutate

        if v_to_mutate == 0:  # endo

            i_v_endo = v_endos_cm_per_s.index(v_endo_cm_per_s)
            v_endos_choice = []

            if i_v_endo != len(v_endos_cm_per_s) - 1:
                v_endos_choice.append(v_endos_cm_per_s[i_v_endo + 1])
            if i_v_endo != 0:
                v_endos_choice.append(v_endos_cm_per_s[i_v_endo - 1])

            if len(v_endos_cm_per_s) == 1:
                v_endos_choice = [v_endo_cm_per_s]

            new_v_endo_cm_per_s, new_v_myo_cm_per_s = random.choice(v_endos_choice), v_myo_cm_per_s

        elif v_to_mutate == 1: # myo
            i_v_myo = v_myos_cm_per_s.index(v_myo_cm_per_s)
            v_myos_choice = []

            if i_v_myo != len(v_myos_cm_per_s) - 1:
                v_myos_choice.append(v_myos_cm_per_s[i_v_myo + 1])
            if i_v_myo != 0:
                v_myos_choice.append(v_myos_cm_per_s[i_v_myo - 1])

            if len(v_myos_cm_per_s) == 1:
                v_myos_choice = [v_myo_cm_per_s]

            new_v_endo_cm_per_s, new_v_myo_cm_per_s = v_endo_cm_per_s, random.choice(v_myos_choice)

        new_v_params = (new_v_endo_cm_per_s, new_v_myo_cm_per_s)

    for _ in range(n_random_mutations):  # Mutations applied to root nodes
        # Root node to be replaced in the existing root_indices
        replace_node_idx = random.randint(0, len(new_root_indices) - 1)

        if random.random() < p_exploration:  # Exploration step (pick root node from all possible root node positions)
            rand_candidate_idx = random.randint(0, len(candidate_root_node_indices) - 1)
            new_root_indices[replace_node_idx] = candidate_root_node_indices[rand_candidate_idx]

        else:  # Exploitation step (mutate to a neighbouring root node)
            replace_mesh_idx = new_root_indices[replace_node_idx]
            x, y, z = xs[replace_mesh_idx], ys[replace_mesh_idx], zs[replace_mesh_idx]
            rand_neighbour = random.choice(candidate_root_neighbours[(x, y, z)])
            new_mesh_idx = grid_dict[rand_neighbour]
            new_root_indices[replace_node_idx] = new_mesh_idx

    new_root_indices.sort()
    params = new_v_params, tuple(new_root_indices)

    return params


def mutate_pop_params(worse_keys, better_keys, all_params, alg, grid_dict, candidate_root_node_indices,
                      candidate_root_neighbours, v_endos_cm_per_s, v_myos_cm_per_s, all_ids_and_diff_scores):
    """ Applies replacement-mutation step to activation parameters of the activation model population

    Args:
        worse_keys (int list): [i_try, ...] corresponding to activation models with worse QRS match
        better_keys (int list): [i_try, ...] corresponding to activation models with better QRS match
        all_params (dict): population params {i_try: (v_endo_param, v_myo_param), (root_idx1, ...)}
        alg (list): alg mesh
        grid_dict (dict): coordinate to mesh idx {(x, y, z): idx, ...}
        candidate_root_node_indices (int list): idxs of allowed root node positions
        candidate_root_neighbours (dict): neighbours of allowed root node posns stored as {(x, y, z): [(x0, y0, z0), ...], ...}
        v_endos_cm_per_s (float list): possible endocardial conduction velocities
        v_myos_cm_per_s (float list): possible myocardial conduction velocities
        all_ids_and_diff_scores (dict): records seen params {param_id: diff_score, iter_no}

    Returns:
        params_copy (dict): mutated population params {i_try: (v_endo_param, v_myo_param), (root_idx1, ...)}
    """
    params_copy = all_params.copy()
    tried_param_ids = set()

    # Replace worse models with random choice of better models (then mutate better models slightly)
    for i, worse_key in enumerate(worse_keys):
        replacement_params = params_copy[random.choice(better_keys)]  # Replace randomly with the better models
        n_mutation_attempts, max_mutation_attempts = 0, 1000

        while True:  # Keep mutating until you find a set of params not tested before
            mutated_replacement_params = mutate_activation_params(replacement_params, alg, grid_dict, candidate_root_node_indices,
                                                candidate_root_neighbours, v_endos_cm_per_s, v_myos_cm_per_s)

            param_id = hash_qrs_param(mutated_replacement_params)

            if param_id not in all_ids_and_diff_scores and param_id not in tried_param_ids:
                tried_param_ids.add(param_id)
                break  # Proceed with this mutated param as it is unseen

            if n_mutation_attempts > max_mutation_attempts:
                raise Exception("Failed to find a mutation that has not been tested before")
            n_mutation_attempts += 1

        params_copy[worse_key] = mutated_replacement_params

    return params_copy


def analyse_sim_qrs_leads(all_electrodes, target_leads, times_sim, times_target, discrepancy_function):
    """ Calc sim leads from electrodes and compare to target leads

        Args:
            all_electrodes (dict[int, np.ndarray]): Dictionary mapping simulation attempt IDs to arrays of pseudo-electrode values.
            target_leads (dict[str, np.ndarray]): Dictionary of target 12-lead ECG waveforms to match against, indexed by lead name.
            times_sim (np.ndarray): Time vector corresponding to simulated pseudo-electrode signals.
            times_target (np.ndarray): Time vector corresponding to the target ECG leads.
            discrepancy_function (Callable): Function that takes (leads_sim, leads_target, times_sim, times_target) and returns a float discrepancy score.

        Returns:
            all_leads_sim (dict[int, dict[str, np.ndarray]]): Dictionary mapping attempt IDs to simulated 12-lead ECGs (lead name → waveform).
            all_diff_scores (dict[int, float]): Dictionary mapping attempt IDs to their corresponding discrepancy scores (rounded to 5 decimal places).
    """
    all_diff_scores, all_leads_sim = {}, {}
    for key, electrodes in all_electrodes.items():
        leads_pseudo = ecg.ten_electrodes_to_twelve_leads(electrodes)
        diff_score = discrepancy_function(leads_pseudo, target_leads)
        all_diff_scores[key] = round(diff_score, 5)
        all_leads_sim[key] = leads_pseudo

    return all_leads_sim, all_diff_scores


def hash_qrs_param(params):
    """ Hashes activation model params to give an identifier

    Args:
        params (tuple tuple): activation model params (v_endo_param, v_myo_param), (root_idx1, ...)

    Returns:
        string: 8 character hash identifying the activation model
    """
    param_str = str(params)
    hash_object = hashlib.md5(param_str.encode())
    return hash_object.hexdigest()[:8]


def ap_heaviside(t, activation_time):
    """ Computes the Heaviside step function at time t

    Args:
    t (float): present time
    activation_time (float): time cell is activated

    Returns:
    1 if t >= activation_time, 0 otherwise
    """
    return np.where(t >= activation_time, 1, 0)


def pseudo_ecg_qrs(times_s, ap_function, electrodes_xyz, elec_grads, dx, activation_cutoff_s, neighbour_arrays,
                   v_params, activation_times_s):
    """Compute pseudo-QRS ∇Vm · ∇(1/r). Can optimise using active/recently active cells

        Args:
            times_s (np.ndarray): Array of time points (in seconds) at which to evaluate the pseudo-ECG.
            ap_function (Callable): Function that returns Vm values given time and activation times.
            electrodes_xyz (np.ndarray): 3D positions of electrodes used to compute pseudo-ECGs (shape: [n_elec, 3]).
            elec_grads (np.ndarray): Precomputed gradients of (1/r) between each electrode and each cell (shape: [3, n_elec, n_cells]).
            dx (float): Spatial resolution (cell side length in microns).
            activation_cutoff_s (float): Time (in seconds) marking the end of the activation phase.
            neighbour_arrays (dict): Dictionary containing neighbor connectivity data for finite difference gradient computation.
            v_params (Optional[np.ndarray]): Array of conduction velocities (cm/s) used to estimate wavefront spread for optimization. If None, no optimization is applied.
            activation_times_s (np.ndarray): Activation times per cell (in seconds).

        Returns:
            electrodes_vs (np.ndarray): Simulated pseudo-ECG voltage signals (shape: [n_elec, len(times_s)]).
    """
    n_elec = len(electrodes_xyz)

    if v_params is not None:
        using_activation_optimisation = True
        iter_dt_s = times_s[1] - times_s[0]
        min_v = min(v_params)
        min_dist_per_timestep_um = min_v * iter_dt_s * 10000
        n_time_steps_to_go_dx = math.ceil(dx / min_dist_per_timestep_um)
        first_activated_t_idxs = np.ceil(activation_times_s / iter_dt_s).astype(int)
    else:
        using_activation_optimisation = False

    unstructured_neighbour_idxs = neighbour_arrays["unstructured_neighbour_idxs"]

    electrodes_vs = np.zeros((n_elec, len(times_s)))

    # Compute ∇Vm at each time point
    for t_idx, time_point_s in enumerate(times_s):
        vms = ap_function(time_point_s, activation_times_s)  # Vms is the activated mask

        # Compute gradients considering which cells have actually been activated already
        if time_point_s <= activation_cutoff_s and using_activation_optimisation:

            t_idxs_to_consider = t_idx - np.arange(n_time_steps_to_go_dx)
            t_idxs_to_consider = t_idxs_to_consider[t_idxs_to_consider >= 0]
            activated_idxs_latest = np.where(np.isin(first_activated_t_idxs, t_idxs_to_consider))[0]
            neighbours_of_active_latest = unstructured_neighbour_idxs[activated_idxs_latest]
            valid_neighbors = neighbours_of_active_latest[neighbours_of_active_latest != -1]
            all_idxs = np.concatenate((activated_idxs_latest, valid_neighbors))

            calc_grad_at_idxs = all_idxs
            calc_grad_at_idxs = np.array(calc_grad_at_idxs, dtype=int)
            grad = ecg.calc_grads(np.array(vms), neighbour_arrays, dx, special_indices=calc_grad_at_idxs)
            grad_x, grad_y, grad_z = grad[:, 0], grad[:, 1], grad[:, 2]
            original_idxs = np.arange(0, len(grad), 1)

        else:  # Compute gradients for all cells
            grad = ecg.calc_grads(np.array(vms), neighbour_arrays, dx)
            grad_x, grad_y, grad_z = grad[:, 0], grad[:, 1], grad[:, 2]
            original_idxs = np.arange(0, len(grad), 1)

        original_idxs = np.array(original_idxs, dtype=int)

        # Dot ∇Vm with ∇(1/r)
        x_comp = grad_x[original_idxs].reshape(-1, 1) * elec_grads[0, :, original_idxs]
        y_comp = grad_y[original_idxs].reshape(-1, 1) * elec_grads[1, :, original_idxs]
        z_comp = grad_z[original_idxs].reshape(-1, 1) * elec_grads[2, :, original_idxs]

        # Sum the components along x, y, z for each electrode (sum over the n_cells dimension)
        electrodes_vs[:, t_idx] = -np.sum(x_comp + y_comp + z_comp, axis=0)
    return electrodes_vs


def compute_batch_ecgs_qrs(pseudo_ecg_function, times_s, ap_function, electrodes_xyz, elec_grads, dx, activation_cutoff_s,
                       neighbour_arrays, batch_indices, batch_v_params, batch_activation_times_s):
    """Compute pseudo-ECGs for a batch of activation scenarios during the QRS phase.

        Args:
            pseudo_ecg_function (Callable): Function used to compute the pseudo-ECG for a single trial.
            times_s (np.ndarray): Array of time points (in seconds) for pseudo-ECG simulation.
            ap_function (Callable): Function returning transmembrane voltage given time and activation times.
            electrodes_xyz (np.ndarray): Positions of electrodes (shape: [n_elec, 3]).
            elec_grads (np.ndarray): Precomputed ∇(1/r) for each electrode-cell pair (shape: [3, n_elec, n_cells]).
            dx (float): Spatial resolution of the grid (in microns).
            activation_cutoff_s (float): Time marking the end of activation (in seconds).
            neighbour_arrays (dict): Dictionary containing neighbour index arrays for gradient calculation.
            batch_indices (Iterable[int]): Indices identifying each trial in the batch.
            batch_v_params (dict): Dictionary mapping trial index to conduction velocity parameters.
            batch_activation_times_s (dict): Dictionary mapping trial index to activation times (per cell, in seconds).

        Returns:
            batch_electrodes (dict): Dictionary mapping trial index to simulated pseudo-ECG (shape per entry: [n_elec, len(times_s)]).
    """
    batch_electrodes = {}
    for i_try in batch_indices:
        batch_electrodes[i_try] = pseudo_ecg_function(times_s, ap_function, electrodes_xyz, elec_grads, dx,
                                                      activation_cutoff_s, neighbour_arrays, batch_v_params[i_try],
                                                      batch_activation_times_s[i_try])
    return batch_electrodes


def get_activation_times(root_indices, all_time_matrix):
    """ Extract activation times from a subset of root nodes.

        Args:
            root_indices (Iterable[int]): Indices of selected root nodes whose activation times will be used.
            all_time_matrix (np.ndarray): Matrix of activation times for all candidate root nodes
                                          (shape: [n_roots_total, n_cells], in milliseconds).

        Returns:
            activation_times_s (np.ndarray): Minimum activation times across selected roots
                                             (shape: [n_cells], in seconds).
    """
    time_matrix = np.vstack([all_time_matrix[root_index] for root_index in root_indices])
    activation_times_s = np.min(time_matrix, axis=0) / 1000  # Convert milliseconds to seconds
    return activation_times_s


def batch_qrs_runner(n_tries, n_per_batch, pseudo_ecg_function, times_s, ap_function, electrodes_xyz, elec_grads,
                     dx, activation_cutoff_s, neighbour_arrays, all_all_time_matrices, qrs_params):
    """ Run pseudo-ECG simulations in batches for multiple activation params

        Args:
            n_tries (int): Total number of trials to simulate.
            n_per_batch (int): Number of trials to include in each batch.
            pseudo_ecg_function (Callable): Function used to compute the pseudo-ECG for a single trial.
            times_s (np.ndarray): Array of time points (in seconds) for pseudo-ECG simulation.
            ap_function (Callable): Function returning transmembrane voltage given time and activation times.
            electrodes_xyz (np.ndarray): Positions of electrodes (shape: [n_elec, 3]).
            elec_grads (np.ndarray): Precomputed ∇(1/r) for each electrode-cell pair (shape: [3, n_elec, n_cells]).
            dx (float): Spatial resolution of the grid (in microns).
            activation_cutoff_s (float): Time marking the end of activation (in seconds).
            neighbour_arrays (dict): Dictionary containing neighbour index arrays for gradient calculation.
            all_all_time_matrices (dict): Dictionary mapping conduction velocity parameters to time matrices
                                          (each matrix shape: [n_roots, n_cells], in milliseconds).
            qrs_params (dict): Dictionary mapping trial index to a tuple:
                               (v_params, root_indices) — where v_params selects the time matrix
                               and root_indices determine the source nodes.

        Returns:
            all_electrodes (dict): Dictionary mapping trial index to simulated pseudo-ECG
                                   (shape per entry: [n_elec, len(times_s)]).
            record_activation_times_s (dict): Dictionary mapping trial index to its computed activation times
                                              (shape per entry: [n_cells], in seconds).
    """
    all_electrodes = {}
    batches = [range(i, min(i + n_per_batch, n_tries)) for i in range(0, n_tries, n_per_batch)]
    batched_activation_times_s = [{} for _ in range(len(batches))]
    batched_v_params = [{} for _ in range(len(batches))]
    record_activation_times_s = {}

    # Precompute all activation times rather than pass in all_time_matrix to each subprocess
    for i, batch in enumerate(batches):
        for i_try in batch:
            # Calculate activation times based on v_params, root_indices and the time matrices
            v_params, root_indices = qrs_params[i_try][0], qrs_params[i_try][1]
            activation_times_s = get_activation_times(root_indices, all_all_time_matrices[v_params])
            batched_v_params[i][i_try] = v_params
            batched_activation_times_s[i][i_try] = activation_times_s
            record_activation_times_s[i_try] = activation_times_s

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_batch_ecgs_qrs, pseudo_ecg_function, times_s, ap_function,
                                   electrodes_xyz, elec_grads, dx, activation_cutoff_s, neighbour_arrays, batch,
                                   batch_v_params, batch_activation_times)

                   for batch, batch_v_params, batch_activation_times in zip(batches, batched_v_params,
                                                                            batched_activation_times_s)]
        # Add electrode outputs to the storage dictionary
        for future in concurrent.futures.as_completed(futures):
            batch_electrodes = future.result()
            all_electrodes.update(batch_electrodes)

    return all_electrodes, record_activation_times_s


def init_roots_and_vels(n_tries, min_n_root_nodes, max_n_root_nodes, candidate_root_node_indices, v_endos_cm_per_s,
                        v_myos_cm_per_s, params_best_guess, use_best_guess):
    """ Initalises population of activation parameters

    Args:
        n_tries (int): population size
        min_n_root_nodes, max_n_root_nodes (int): min and max allowed number of root nodes of individual activation models
        candidate_root_node_indices (int list): idxs of allowed root node positions
        v_endos_cm_per_s (float list): possible endocardial conduction velocities
        v_myos_cm_per_s (float list): possible myocardial conduction velocities
        params_best_guess (tuple tuple): for use when simulating just 1 activation model
        use_best_guess (bool): simulates just 1 activation model using params_best_guess if True

    Returns:
        current_iter_params (dict): population params {i_try: (v_endo_param, v_myo_param), (root_idx1, ...)}
    """
    current_iter_params = {}

    if use_best_guess:  # Set the zeroth parameter combo to the best guess from before
        root_indices_mesh = list(params_best_guess[1])
        v_params = params_best_guess[0]
        current_iter_params[0] = (v_params, tuple(root_indices_mesh))
        return current_iter_params

    # Prepare initial root nodes
    for i_try in range(n_tries):
        n_root_nodes = random.randint(min_n_root_nodes, max_n_root_nodes)
        root_indices_mesh = []

        for _ in range(n_root_nodes):  # Select n random root nodes from candidate_root_node_indices
            # TODO handling of duplicate root node indices
            rand_candidate_idx = random.randint(0, len(candidate_root_node_indices) - 1)
            root_indices_mesh.append(candidate_root_node_indices[rand_candidate_idx])

        # Select random v_endo_param, v_myo_param from the possible parameter sets
        v_endo_param, v_myo_param = random.choice(v_endos_cm_per_s), random.choice(v_myos_cm_per_s)
        v_params = (v_endo_param, v_myo_param)

        root_indices_mesh.sort()  # Prevent selection order mattering
        current_iter_params[i_try] = (v_params, tuple(root_indices_mesh))

    return current_iter_params


def create_sparse_adjacency_time(adjacency_list, v_fibers_cm_per_s, v_sheets_cm_per_s, v_normals_cm_per_s,
                                   v_endo_cm_per_s, endo_mask, use_fibers, fibers=None, sheets=None, normals=None):
    """ Creates sparse adjacency matrix representing travel times between cells using fast endo and myofibers

    Args:
        adjacency_list (dict): keys are cell indices and values are lists of tuples. Each tuple contains a neighboring
                               cell index and the displacement vector to that neighbor.
        v_fibers_cm_per_s, v_sheets_cm_per_s, v_normals_cm_per_s (floats): f, s, n conduction velocities
        v_endo_cm_per_s (float): isotropic endocardial conduction velocity
        endo_mask (bool array): flags endocardial cells in the alg mesh
        use_fibers (bool): uses fibers, sheets, normals vectors if True, defaults myo conduction to v_fibers if False
        fibers, sheets, normals (array of floats tuples): fiber, sheet, normal vectors for each cell

    Returns:
        Sparse adjacency matrix where each entry is of form (travel_time_ms, (idx, neighbor_idx))
    """
    row_indices, col_indices, data = [], [], []
    um_to_cm = 1e-4
    s_to_ms = 1000
    n_cells = len(adjacency_list)

    for idx, neighbors in adjacency_list.items():
        for neighbour_idx, displacement in neighbors:

            distance_um = np.linalg.norm(displacement)

            # Keep in mind 4 possibilities: endo-endo, endo-myo, myo-endo, myo-myo
            if endo_mask[idx] and endo_mask[neighbour_idx]:  # Isotropic fast endocardial propagation
                v_total_cm_per_s = v_endo_cm_per_s  # endo-endo

            elif use_fibers:  # Anisotropic bulk myocardial propagation using fibers
                disp_normed = displacement / distance_um
                f_vec, s_vec, n_vec = fibers[idx], sheets[idx], normals[idx]
                f_proj, s_proj, n_proj = np.dot(disp_normed, f_vec), np.dot(disp_normed, s_vec), np.dot(disp_normed, n_vec)
                v_f_proj, v_s_proj, v_n_proj = v_fibers_cm_per_s * np.abs(f_proj), v_sheets_cm_per_s * np.abs(s_proj), v_normals_cm_per_s * np.abs(n_proj)
                v_total_cm_per_s = np.sqrt(v_f_proj**2 + v_s_proj**2 + v_n_proj**2)

            else:  # Bulk myocardial propagation (no fibers, v_myo=v_fibers)
                v_total_cm_per_s = v_fibers_cm_per_s

            distance_cm = distance_um * um_to_cm
            travel_time_ms = distance_cm / v_total_cm_per_s * s_to_ms

            row_indices.append(idx)
            col_indices.append(neighbour_idx)
            data.append(travel_time_ms)

    return csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_cells))


def compute_time_matrix_batch(batch_args):
    """Compute time matrices for a batch of mesh arguments.

        Args:
            batch_args (list of tuples): Each tuple contains:
                - v_endo (array-like): endo conduction velocity
                - v_myo (array-like): myo conduction velocity
                - adjacency_list_26 (list): Adjacency info for 26-neighbour connectivity.
                - endo_mask (array-like): endocardial mask
                - use_fibers (bool): whether to use fiber velocities (not supported)
                - candidate_root_node_indices (list): Mesh idxs of candidate root nodes.

        Returns:
            dict: Keys are tuples (v_endo, v_myo), values are dicts mapping mesh index to
                  time arrays computed from candidate root nodes.
    """
    results = {}
    for args in batch_args:
        (v_endo, v_myo, adjacency_list_26, endo_mask, use_fibers, candidate_root_node_indices) = args
        v_fibers, v_sheets, v_normals = v_myo, v_myo, v_myo
        adj_matrix = create_sparse_adjacency_time(
            adjacency_list_26, v_fibers, v_sheets, v_normals, v_endo, endo_mask, use_fibers
        )
        all_time_matrix = dijkstra(adj_matrix, indices=candidate_root_node_indices, return_predecessors=False)
        times_all_candidate_root_nodes = {
            mesh_idx: all_time_matrix[i].astype(np.float32)
            for i, mesh_idx in enumerate(candidate_root_node_indices)
        }
        results[(v_endo, v_myo)] = times_all_candidate_root_nodes
    return results


def batcher(iterable, batch_size):
    """Yield successive batches from an iterable.

        Args:
            iterable (iterable): The data source to be split into batches.
            batch_size (int): Number of items per batch.

        Yields:
            list: Next batch of items from the iterable.
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


def prepare_target_leads(leads_targ_in, iter_dt_s, qrs_safety_s):
    """ Convert leads target format {"I": [times, signals], ...} to {"I": signals, ...} and times_target_s + pads

        Args:
            leads_targ_in (dict): Dictionary of target ECG leads, where each entry is of the form:
                                  {"LeadName": [times_array, signal_array]}, with times in seconds.
            iter_dt_s (float): Simulation time step (in seconds).
            qrs_safety_s (float): Additional time padding added to ensure the QRS window is fully captured.

        Returns:
            leads_target (dict): Dictionary mapping lead names to padded and resampled ECG signals
                                 (shape per entry: [len(times_s)]).
            times_target_s (np.ndarray): Time points corresponding to the target ECG signals (in seconds).
            times_s (np.ndarray): Time points used in the simulation (aligned with iter_dt_s).
            total_time_s (float): Total duration of the padded time window (in seconds).
    """

    lead_names_targ_in = list(leads_targ_in.keys())
    leads_target = {name: leads_targ_in[name][1] for name in lead_names_targ_in}
    times_target_s = np.round(leads_targ_in[lead_names_targ_in[0]][0], 6)

    dts_target = np.diff(times_target_s)
    if not np.allclose(dts_target, dts_target[0], atol=1e-7):
        raise Exception(f"Inconsistent time steps: {dts_target}")
    iter_dt_targ_s = dts_target[0]

    max_t_targ_s = np.max(times_target_s)
    times_target_s = list(times_target_s)
    n_pre_safety = len(times_target_s)

    # Expanding times_target to include time padding safety factor
    for t_clin in np.arange(max_t_targ_s + iter_dt_targ_s, max_t_targ_s + qrs_safety_s, iter_dt_targ_s):
        times_target_s.append(t_clin)
    times_target_s = np.array(times_target_s)
    n_post_safety = len(times_target_s)

    for lead_name in leads_target.keys():  # Zero padding to leads signals
        for _ in range(n_post_safety - n_pre_safety):
            leads_target[lead_name] = np.append(leads_target[lead_name], 0.0)

    qrs_off_s = max(times_target_s)  # TODO avoid this repetition for clin vs. sim target QRSes
    total_time_s = qrs_off_s
    times_s = np.round(np.arange(0, total_time_s + iter_dt_s, iter_dt_s), decimals=6)
    times_s = times_s[times_s <= total_time_s]  # Prevent overstepping beyond total_time_s

    times_target_s, leads_target = ecg.match_times(times_target_s, leads_target, times_s)

    return leads_target, times_target_s, times_s, total_time_s


def find_optimal_scaling(leads_a, leads_b):
    """Compute optimal non-negative scaling factor to align one set of ECG signals to another.

        Args:
            leads_a (dict): Dictionary of simulated or input ECG signals (shape per lead: [n_timepoints]).
            leads_b (dict): Dictionary of target ECG signals to compare against (same shape and lead names as `leads_a`).

        Returns:
            alpha (float): Optimal non-negative scaling factor to apply to `leads_a`
                           to minimize squared error with respect to `leads_b`.
    """
    for lead in leads_a:  # Sanity check that leads are comparable, and have same shape
        if lead not in leads_b:
            raise ValueError(f"Lead '{lead}' is missing in leads_b.")
        if leads_a[lead].shape != leads_b[lead].shape:
            raise ValueError(f"Shape mismatch for lead '{lead}': {leads_a[lead].shape} vs {leads_b[lead].shape}")

    leads_a_signal = np.concatenate([leads_a[lead] for lead in leads_a])
    leads_b_signal = np.concatenate([leads_b[lead] for lead in leads_b])
    numerator = np.sum(leads_a_signal * leads_b_signal)
    denominator = np.sum(leads_a_signal ** 2)
    alpha = numerator / denominator if denominator != 0 else 0
    alpha = max(alpha, 0)  # Note possible issue of alpha < 0 in cases of anticorrelated signals

    return alpha


def calc_discrepancy(leads_sim, leads_target):
    """Calculate mean relative discrepancy between simulated and target ECG leads using 1 optimal scaling.

        Args:
            leads_sim (dict): Dictionary of simulated ECG signals (shape per lead: [n_timepoints]).
            leads_target (dict): Dictionary of target ECG signals (same shape and lead names as `leads_sim`).

        Returns:
            mean_discrepancy (float): Mean of the relative discrepancies across all leads.
                                      Each discrepancy is computed as the average absolute error
                                      normalized by the target QRS amplitude.

        Notes:
            - A single optimal non-negative scaling factor is applied to all simulated leads
              to best align them (in least-squares sense) with the target leads.
            - The discrepancy for each lead is normalized by its QRS amplitude (peak-to-peak).
            - This metric emphasizes shape similarity rather than absolute amplitude.
    """
    lead_names = list(leads_sim.keys())
    alpha = find_optimal_scaling(leads_sim, leads_target)  # Optimal scaling to allow us to compare the 2 leads
    leads_sim_rescaled = {name: leads_sim[name] * alpha for name in lead_names}
    target_qrs_amps = {name: np.max(leads_target[name]) - np.min(leads_target[name]) for name in lead_names}
    leads_discrepancy = {name: np.mean(np.abs(leads_sim_rescaled[name] - leads_target[name]) / target_qrs_amps[name])
                          for name in lead_names}
    leads_discrepancy_overlay = list(leads_discrepancy.values())
    return np.mean(leads_discrepancy_overlay)


def calc_discrepancy_separate_scaling(leads_sim, leads_target):
    """Calculate mean relative discrepancy between simulated and target ECG leads using separate scaling
       for limb and precordial leads.

        Args:
            leads_sim (dict): Dictionary of simulated ECG signals (shape per lead: [n_timepoints]).
            leads_target (dict): Dictionary of target ECG signals (same shape and lead names as `leads_sim`).

        Returns:
            mean_discrepancy (float): Mean of the relative discrepancies across all leads.
                                      Each discrepancy is computed as the average absolute error
                                      normalized by the target QRS amplitude.
    """
    limb_leads, precordial_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'], ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    limb_leads_present = [l for l in limb_leads if l in leads_sim]
    precordial_leads_present = [l for l in precordial_leads if l in leads_sim]

    alpha_limb = find_optimal_scaling({l: leads_sim[l] for l in limb_leads_present},
                                      {l: leads_target[l] for l in limb_leads_present}) if limb_leads_present else 1.0

    alpha_prec = find_optimal_scaling({l: leads_sim[l] for l in precordial_leads_present},
                                      {l: leads_target[l] for l in precordial_leads_present}) if precordial_leads_present else 1.0

    leads_sim_rescaled = {}  # Rescale simulated leads
    for lead in leads_sim:
        if lead in limb_leads_present:
            leads_sim_rescaled[lead] = leads_sim[lead] * alpha_limb
        elif lead in precordial_leads_present:
            leads_sim_rescaled[lead] = leads_sim[lead] * alpha_prec
        else:  # If lead is neither limb nor precordial, keep original scaling
            leads_sim_rescaled[lead] = leads_sim[lead]

    target_qrs_amps = {name: np.max(leads_target[name]) - np.min(leads_target[name]) for name in leads_sim_rescaled}
    leads_discrepancy = {name: np.mean(np.abs(leads_sim_rescaled[name] - leads_target[name]) / target_qrs_amps[name])
                        for name in leads_sim_rescaled}
    leads_discrepancy_overlay = list(leads_discrepancy.values())

    return np.mean(leads_discrepancy_overlay)


def calc_discrepancy_shape_only_old(leads_sim, leads_target):

    lead_names_to_compare = LEAD_NAMES_12

    # Rescaling
    amps_leads_sim = {key: max(val) - min(val) for key, val in leads_sim.items()}
    rescaled_leads_sim = {key: val / amps_leads_sim[key] for key, val in leads_sim.items()}

    amps_leads_target = {key: max(val) - min(val) for key, val in leads_target.items()}
    rescaled_leads_target = {key: val / amps_leads_target[key] for key, val in leads_target.items()}

    lead_diffs = {}

    for i, key in enumerate(rescaled_leads_sim.keys()):

        if key in rescaled_leads_target:  # Only compare leads in both target and pseudo ECG

            lead_a, lead_b = rescaled_leads_sim[key], rescaled_leads_target[key]
            size_a, size_b = np.max(lead_a) - np.min(lead_a), np.max(lead_b) - np.min(lead_b)

            if not np.isclose(size_a, size_b, atol=0, rtol=1e-4):
                raise Exception("Amplitudes of the leads being compared is different")

            diffs = np.abs(lead_a - lead_b)

            if lead_names_to_compare is not None:
                if key in lead_names_to_compare:  # To just compare specific leads
                    lead_diffs[key] = sum(diffs) / len(lead_a)
            else:
                lead_diffs[key] = sum(diffs) / len(lead_a)  # Mean diffs per sample

    #times = [i for i in range(len(leads_sim["I"]))]
    #ecg2.plot_ecg([times, times], [rescaled_leads_sim, rescaled_leads_target], colors=["red", "black"])

    diff_score = sum(lead_diffs.values()) / len(lead_names_to_compare)
    #print(f"{diff_score=}")

    return diff_score


def mesh_subset_with_dist_constraint(alg, dist_limit_um, neighbour_dist_um):
    """ Deterministic sampling of alg cells where sampled points must be > dist_limit_um from all other sample
        points

        Args:
            alg (list): alg mesh
            dist_limit_um (float): mesh cells can only be sampled further than dist_limit from all other sampled cells
            neighbour_dist_um (float): neighbouring cells of sampled cells are stored as neighbours within this distance

        Returns:
            points_subset (list of float tuples): sampled points [(x0, y0, z0), (x1, y1, z1), ...]
            points_neighbours (dict): neighbours of points_subset stored as key (x, y, z): [(x0, y0, z0), ...]
    """
    # TODO: KD-trees reimplementation
    xs, ys, zs, *_ = alg_utils.unpack_alg_geometry(alg)
    grid_coarse = {}
    neighbours = np.concatenate((np.array([(0, 0, 0)]), NEIGHBOURS_26))  # Include same cell at origin
    # grid_scale_um must always be >= dist_limit_um to ensure dist limit is properly enforced
    grid_scale_um = dist_limit_um
    points_subset = []

    for x_cand, y_cand, z_cand in zip(xs, ys, zs):

        # Indices of the coarse grid which will contain the random point
        i_crs, j_crs, z_crs = int(x_cand // grid_scale_um), int(y_cand // grid_scale_um), int(z_cand // grid_scale_um)

        add_to_mesh = 1

        # Check neighbouring chunks for other sampled points
        for di, dj, dk in neighbours:
            ni, nj, nk = i_crs + di, j_crs + dj, z_crs + dk

            if (ni, nj, nk) in grid_coarse:
                # Check all points contained in this coarse grid entry
                for (x_other, y_other, z_other) in grid_coarse[(ni, nj, nk)]:
                    dist_sq = (x_cand - x_other) ** 2 + (y_cand - y_other) ** 2 + (z_cand - z_other) ** 2

                    # Refuse to add it to the mesh if it is too close to another point and stop checking this neighbour
                    if dist_sq <= dist_limit_um ** 2:
                        add_to_mesh = 0
                        break
            # If too close to another point, stop checking all neighbours as this candidate point is already rejected
            if not add_to_mesh:
                break

        # Add the successful candidate point to the coarse grid
        if add_to_mesh:

            # Initialise this coarse grid element if it does not yet exist
            if (i_crs, j_crs, z_crs) not in grid_coarse:
                grid_coarse[(i_crs, j_crs, z_crs)] = []

            grid_coarse[(i_crs, j_crs, z_crs)].append((x_cand, y_cand, z_cand))
            points_subset.append((x_cand, y_cand, z_cand))

    # New coarse grid suitable for distance checks up to neighbour_dist_um, formed from accepted the points subset
    grid_coarse = {}
    for x, y, z in points_subset:

        i_crs, j_crs, z_crs = int(x // neighbour_dist_um), int(y // neighbour_dist_um), int(z // neighbour_dist_um)

        # Initialise this coarse grid element if it does not yet exist
        if (i_crs, j_crs, z_crs) not in grid_coarse:
            grid_coarse[(i_crs, j_crs, z_crs)] = []

        # Add accepted point from points subset into the coarse grid
        grid_coarse[(i_crs, j_crs, z_crs)].append((x, y, z))

    # Neighbours of point (key (x, y, z)) stored as list of tuples [(x0, y0, z0), ...]
    points_neighbours = {}

    for x, y, z in points_subset:

        # Initialise neighbour list for this point
        points_neighbours[(x, y, z)] = []

        # Indices of the coarse grid which contains point (x, y, z)
        i_crs, j_crs, z_crs = int(x // neighbour_dist_um), int(y // neighbour_dist_um), int(z // neighbour_dist_um)

        # Check neighbouring chunks for other points
        for di, dj, dk in neighbours:
            ni, nj, nk = i_crs + di, j_crs + dj, z_crs + dk

            if (ni, nj, nk) in grid_coarse:
                # Check all points contained in this coarse grid entry
                for (x_other, y_other, z_other) in grid_coarse[(ni, nj, nk)]:
                    dist_sq = (x - x_other) ** 2 + (y - y_other) ** 2 + (z - z_other) ** 2

                    # (x_other, y_other, z_other) is classed as a neighbour if within this distance
                    if dist_sq <= neighbour_dist_um ** 2:
                        points_neighbours[(x, y, z)].append((x_other, y_other, z_other))

    return points_subset, points_neighbours