import numpy as np


def compute_speed_PCA(links_positions, links_vel, sim_fraction=1.0):
    """Compute axial and lateral speed based on the PCA of the link positions.

    At each time step the principal axis of the body (largest eigenvector of
    the 2-D position covariance matrix) is aligned head→tail and used to
    project the mean COM velocity into forward and lateral components.

    Parameters
    ----------
    links_positions : array (nt, n_links, 3)
    links_vel       : array (nt, n_links, 3)  — only xy components used
    sim_fraction    : float, optional
        Fraction of time steps to consider (from the end).  Default 1.0.

    Returns
    -------
    speed_forward : list[float], length = nt * sim_fraction
    speed_lateral : list[float], length = nt * sim_fraction
    """
    n_steps            = links_positions.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)

    links_pos_xy  = links_positions[-n_steps_considered:, :, :2]
    joints_vel_xy = links_vel[-n_steps_considered:, :, :2]
    time_idx      = links_pos_xy.shape[0]

    speed_forward = []
    speed_lateral = []

    for idx in range(time_idx):
        x = links_pos_xy[idx, :, 0]
        y = links_pos_xy[idx, :, 1]

        pheadtail = links_pos_xy[idx][0] - links_pos_xy[idx][-1]  # head - tail
        vcom_xy   = np.mean(joints_vel_xy[idx], axis=0)

        covmat               = np.cov([x, y])
        eig_values, eig_vecs = np.linalg.eig(covmat)
        largest_index        = np.argmax(eig_values)
        largest_eig_vec      = eig_vecs[:, largest_index]

        ht_direction    = np.sign(np.dot(pheadtail, largest_eig_vec))
        largest_eig_vec = ht_direction * largest_eig_vec

        v_com_forward_proj = np.dot(vcom_xy, largest_eig_vec)

        left_pointing_vec = np.cross(
            [0, 0, 1],
            [largest_eig_vec[0], largest_eig_vec[1], 0],
        )[:2]

        v_com_lateral_proj = np.dot(vcom_xy, left_pointing_vec)

        speed_forward.append(v_com_forward_proj)
        speed_lateral.append(v_com_lateral_proj)

    return speed_forward, speed_lateral
