import numpy as np
import utils2
import os
from scipy.spatial.transform import Rotation as R
import cache2


def read_alg_mesh(file_path):
    """ Loads alg file of form x, y, z, dx, dy, dz, fields, note geometry values all ints

    Args:
        file_path (string): path to .alg file to load

    Returns:
        alg (list of arrays): alg in format [xs, ys, zs, dxs, dys, zs, fields1, ...]
    """
    with open(file_path, 'r') as alg_file:
        # Initialise alg list structure
        first_line = alg_file.readline()
        fields = [f.strip() for f in first_line.split(",")]
        #fields = first_line.split(",")
        alg = [[] for _ in range(len(fields))]

        # Process first line
        for i in range(len(alg)):
            alg[i].append(utils2.safe_float(fields[i]))

        # Process the rest of the file
        for line in alg_file:
            fields = [f.strip() for f in line.split(",")]

            for i in range(len(alg)):
                alg[i].append(utils2.safe_float(fields[i]))

    alg = [np.array(alg_entry) for alg_entry in alg]

    # Geometry x, y, z, dx, dy, dz stored as int64
    for i in range(6):
        alg[i] = alg[i].astype(np.int64)

    return alg


def unpack_alg_geometry(alg):
    """ Get mesh cell coords and dxs from the loaded alg

    Args:
        alg (list of arrays): alg in format [xs, ys, zs, dxs, dys, zs, fields1, ...]

    Returns:
        xs, ys, zs, dxs, dys, dzs (tuple of arrays)
    """
    return alg[0], alg[1], alg[2], alg[3], alg[4], alg[5]


def get_dx(xs):
    """ Compute dx from xs

    Args:
        xs (array): mesh coordinates along one axis

    Returns:
        dx (int): spatial discretisation along this axis
    """
    if not len(xs):
        raise Exception(f"{xs=} cannot calculate dx")
    uniques = np.unique(np.diff(np.sort(np.unique(xs))))
    dx = uniques[0]

    if len(uniques) > 1:
        raise Exception(f"Get dx failing: unique dxs {uniques}")


    return int(dx)


def alg_from_xs(xs, ys, zs, fields=None, dx=None):
    """ Convert xs, ys, zs to alg list

    Args:
        xs, ys, zs (arrays): mesh coordinates along the 3 axes
        fields (list of arrays): Optionally include fields in the alg
        dx (int): Optionally pre-specify dx or it will be computed

    Returns:
        alg (list of arrays): alg in format [xs, ys, zs, dxs, dys, zs, fields1, ...]
    """
    if dx is None:
        dx = get_dx(xs)
    lxs = np.array([dx / 2 for _ in range(len(xs))])
    alg = [xs, ys, zs, lxs, lxs, lxs]

    if fields is not None:
        for field in fields:
            alg.append(field)

    return alg


def check_orthonormal_basis(a, b, c, atol=1e-8):
    # Check unit length
    lengths_ok = all(np.isclose(np.linalg.norm(v), 1.0, atol=atol) for v in [a, b, c])

    # Check orthogonality
    orthogonal_ok = (np.isclose(np.dot(a, b), 0.0, atol=atol) and
                     np.isclose(np.dot(a, c), 0.0, atol=atol) and
                     np.isclose(np.dot(b, c), 0.0, atol=atol))

    return lengths_ok and orthogonal_ok


def rotate_electrodes(electrodes_xyz, axis0, axis1, axis2, axis_to_use, run_dir, angle_rot_deg, varying_angle,
                      center_of_mass):

    if not (angle_rot_deg != 0 and varying_angle):
        print("No electrode rotation")
        return electrodes_xyz

    orthonormal = check_orthonormal_basis(axis0, axis1, axis2)
    if not orthonormal:
        raise ValueError(f"Rotation axes not orthonormal")

    print(f"Changing electrode positions by {angle_rot_deg} degrees")

    print(f"{electrodes_xyz=}")
    electrodes_xyz_rot, *_ = rotate_electrodes_xyz(electrodes_xyz, center_of_mass, axis_to_use, angle_rot_deg)
    print(f"{electrodes_xyz_rot=}")
    print(f"{electrodes_xyz=}")

    xs_e, ys_e, zs_e = [e[0] for e in electrodes_xyz], [e[1] for e in electrodes_xyz], [e[2] for e in electrodes_xyz]
    alg_elecs = alg_from_xs(xs_e, ys_e, zs_e, dx=5000)
    save_alg_mesh(f"{run_dir}/electrodes_xyz_pre_rot.alg", alg_elecs)

    xs_r, ys_r, zs_r = [e[0] for e in electrodes_xyz_rot], [e[1] for e in electrodes_xyz_rot], [e[2] for e in electrodes_xyz_rot]
    alg_elecs_rot = alg_from_xs(xs_r, ys_r, zs_r, dx=5000)
    save_alg_mesh(f"{run_dir}/electrodes_xyz_rot.alg", alg_elecs_rot)

    np.save(f"{run_dir}/electrodes_xyz_rot.npy", electrodes_xyz_rot)

    electrodes_xyz = electrodes_xyz_rot

    return electrodes_xyz


def rotate_electrodes_xyz(electrodes_xyz, center_of_rotation, axis, angle_deg):

    if not np.isclose(np.linalg.norm(axis), 1.0):
        raise Exception("Why is the rotation axis not normalised when rotating electrodes?")

    xs_elec, ys_elec, zs_elec = [e[0] for e in electrodes_xyz], [e[1] for e in electrodes_xyz], [e[2] for e in electrodes_xyz]
    elec_pts = np.vstack([xs_elec, ys_elec, zs_elec]).T
    elec_pts_centered = elec_pts - center_of_rotation

    rotation = R.from_rotvec(np.deg2rad(angle_deg) * axis)
    rotated_pts = rotation.apply(elec_pts_centered) + center_of_rotation
    xs_elec_rot, ys_elec_rot, zs_elec_rot = rotated_pts.T

    electrodes_xyz_rot = [tuple(pt) for pt in rotated_pts]

    return electrodes_xyz_rot, xs_elec_rot, ys_elec_rot, zs_elec_rot


def radially_translate_electrodes_xyz(electrodes_xyz, center_of_mass, rad_translation_um, elec_idxs_to_translate):
    elec_vecs = [elec - center_of_mass for elec in electrodes_xyz]
    elec_unit_vecs = [elec_vec / np.linalg.norm(elec_vec) for elec_vec in elec_vecs]
    electrodes_xyz_rad_translated = electrodes_xyz.copy()

    for i in elec_idxs_to_translate:
        elec_xyz, elec_unit_vec = electrodes_xyz[i], elec_unit_vecs[i]
        elec_xyz_new = elec_xyz + elec_unit_vec * rad_translation_um
        electrodes_xyz_rad_translated[i] = elec_xyz_new
    return electrodes_xyz_rad_translated


def translate_electrodes(electrodes_xyz, elec_rad_translation_um, elec_idxs_to_translate, center_of_mass, run_dir):
    if elec_rad_translation_um == 0:
        print("No electrode translation")
        electrodes_xyz_alg = electrodes_xyz_to_alg(electrodes_xyz)
        save_alg_mesh(f"{run_dir}/electrodes_xyz_pre_translated.alg", electrodes_xyz_alg)
        return electrodes_xyz

    print(f"Changing {elec_idxs_to_translate} electrode positions by {elec_rad_translation_um} um")

    electrodes_xyz_rad_translated = radially_translate_electrodes_xyz(electrodes_xyz, center_of_mass,
                                                                                elec_rad_translation_um,
                                                                                elec_idxs_to_translate)
    electrodes_xyz_alg = electrodes_xyz_to_alg(electrodes_xyz)
    electrodes_xyz_rad_trans_alg = electrodes_xyz_to_alg(electrodes_xyz_rad_translated)

    save_alg_mesh(f"{run_dir}/electrodes_xyz_pre_translated.alg", electrodes_xyz_alg)
    save_alg_mesh(f"{run_dir}/electrodes_xyz_translated.alg", electrodes_xyz_rad_trans_alg)

    np.save(f"{run_dir}/electrodes_xyz_rad.npy", electrodes_xyz)

    electrodes_xyz = electrodes_xyz_rad_translated

    return electrodes_xyz


def electrodes_xyz_to_alg(electrodes_xyz, dx=5000, fields=None):
    xs, ys, zs = [e[0] for e in electrodes_xyz], [e[1] for e in electrodes_xyz], [e[2] for e in electrodes_xyz]
    electrodes_xyz_alg = alg_from_xs(xs, ys, zs, dx=dx, fields=fields)
    return electrodes_xyz_alg


def save_alg_mesh(path, alg, remove_old=True):
    """ Save alg with any number of fields to a file.

    Args:
        path (str): Path to the file where the mesh data will be saved.
        alg (list of arrays): alg in format [xs, ys, zs, dxs, dys, zs, fields1, ...]
        remove_old (bool): Flag to indicate if an existing file should be removed before saving.
    """

    # TODO rounding of values to stop large .alg files

    # Ensure alg contains xs, ys, zs, dx, dx, dx
    if len(alg) < 6:
        raise Exception(f"Trying to save alg with only {len(alg)} fields")

    # Cast entries to numpy arrays with a warning
    for i, field in enumerate(alg):
        if not isinstance(field, np.ndarray):
            alg[i] = np.array(field)
            #print(f"Alg field {i} casted to numpy array")

    # Create directory if it does not yet exist
    alg_dir = os.path.dirname(path)
    if not os.path.exists(alg_dir):
        os.makedirs(alg_dir)

    # Checking if file already exists
    utils2.handle_preexisting_path(path, remove_old)

    field_lengths = []
    # Checks for boolean arrays in the alg and converts them to int to avoid saving True/False in the .alg
    for i, arr in enumerate(alg):
        field_lengths.append(len(arr))
        if arr.dtype == bool:
            alg[i] = arr.astype(int)

    if len(set(field_lengths)) != 1:
        raise ValueError(f"Uneven number of cells between alg fields! Check {field_lengths=}")

    n_cells = len(alg[0])
    n_fields = len(alg)

    # If no cells to save, don't try to save the mesh
    if n_cells <= 0:
        print("No cells to save")
        return -1

    # Determine the maximum column width for each field
    max_col_widths = [max(len(str(item)) for item in field) for field in alg]

    with open(path, 'a') as alg_file:
        for i in range(n_cells):
            # Build each line with appropriate formatting
            line_parts = []
            for p in range(n_fields):
                value = str(alg[p][i]) + ","
                padded_value = value.ljust(max_col_widths[p] + 2)
                line_parts.append(padded_value)
            line = ''.join(line_parts).rstrip()  # Remove any trailing spaces for the last column
            alg_file.write(line[:-1] + "\n")  # Remove final comma and add newline


def make_grid_dictionary(xs, ys, zs, values=None):
    """ Store mesh coordinates as hash map

    Args:
        xs, ys, zs (arrays of floats): coordinates of mesh cell centres
        values (array): optional alternative to using indices as the value stored in dict

    Returns:
        grid_dict (dict): key (x, y, z) mapped onto original index (OR optionally values) of point in the mesh
    """
    if values is None:  # coord : original index
        grid_dict = {(x, y, z): idx for idx, (x, y, z) in enumerate(zip(xs, ys, zs))}
    else:  # coord : value
        grid_dict = {(x, y, z): val for x, y, z, val in (zip(xs, ys, zs, values))}

    return grid_dict