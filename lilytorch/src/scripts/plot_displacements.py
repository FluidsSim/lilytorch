
import h5py
from farms_core.sensors.sensor_convention import sc
import numpy as np
import matplotlib.pyplot as plt
import os


dir='/data/andreaferrario/ns_data/2026-01-06T10:35:02.292356/'
file = dir + 'simulation.hdf5'


def _get_distance_to_quadratic(
    point       : np.ndarray[float],
    coefficients: np.ndarray[float],
) -> tuple[float, np.ndarray[float]]:
    ''' Find the point on a parabola where the distance to a given point is minimized '''

    # Find roots of the derivative of the squared distance between point and parabola
    x0, y0  = point
    a, b, c = coefficients

    squared_distance_derivative_coeff = [
        2 * a**2,
        3 * a * b,
        1 + b**2 + 2 * a * (c-y0),
        b * (c-y0) - x0,
    ]

    x_roots = np.roots(squared_distance_derivative_coeff)

    # Find the minimum distance
    min_signed_dist = np.inf
    x_min, y_min    = None, None
    f_y0            = np.polyval(coefficients, x0)

    for x_root in x_roots:

        if not np.isreal(x_root):
            continue

        x_root   = np.real(x_root)
        y_root   = np.polyval(coefficients, x_root)
        distance = np.sqrt((x_root - point[0])**2 + (y_root - point[1])**2)

        # Compute signed distance
        signed_dist = distance * np.sign(f_y0 - y_root)

        # Update candidate minimum
        if abs(signed_dist) < abs(min_signed_dist):
            min_signed_dist = signed_dist
            x_min, y_min    = x_root, y_root

    return min_signed_dist, np.array([x_min, y_min])


def _compute_body_linear_fit_all(
    coordinates_xy: np.ndarray,
    n_links_pca   : int = None,
) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute the PCA of the links positions at all steps '''

    flattened_coordinates = coordinates_xy[:, :n_links_pca].reshape(-1, 2)
    cov_mat = np.cov(
        [
            flattened_coordinates[:, 0],
            flattened_coordinates[:, 1],
        ]
    )

    eig_values, eig_vecs = np.linalg.eig(cov_mat)
    largest_index        = np.argmax(eig_values)
    direction_fwd        = eig_vecs[:, largest_index]

    # Align the direction with the start-finish axis
    p_start2end    = coordinates_xy[-1, 0] - coordinates_xy[0, 0]
    direction_sign = np.sign( np.dot( p_start2end, direction_fwd ) )
    direction_fwd  = direction_sign * direction_fwd

    direction_left = np.cross(
        [0,0,1],
        [direction_fwd[0], direction_fwd[1], 0]
    )[:2]

    return direction_fwd, direction_left

def _transform_coordinates_to_body_frame(
    coordinates_xy: np.ndarray,
    n_links_pca   : int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ''' Transform coordinates to the body frame, defined by the PCA of the links positions '''

    direction_fwd, direction_left = _compute_body_linear_fit_all(
        coordinates_xy = coordinates_xy,
        n_links_pca    = n_links_pca,
    )

    if np.all( direction_fwd == [0,0] ) and np.all( direction_left == [0,0] ):
        # No transformation possible, keep the original coordinates
        return coordinates_xy, np.array([1,0]), np.array([0,1])

    coordinates_transformed = np.zeros_like(coordinates_xy)
    for step in range(coordinates_xy.shape[0]):
        coordinates_transformed[step, :, 0] = np.dot(coordinates_xy[step], direction_fwd)
        coordinates_transformed[step, :, 1] = np.dot(coordinates_xy[step], direction_left)

    return coordinates_transformed, direction_fwd, direction_left


def _compute_links_displacements(
    coordinates_xy: np.ndarray[float],
    n_links_pca   : int = None,
) -> dict[str, np.ndarray[float]]:
    ''' Compute the displacement of the links from the fitted parabola'''

    # Convert to body axis coordinates
    (
        coordinates_xy_transformed,
        direction_fwd,
        direction_left,
    ) = _transform_coordinates_to_body_frame(
        coordinates_xy = coordinates_xy,
        n_links_pca    = n_links_pca,
    )

    # Polynomial fitting in body axis coordinates
    flattened_coordinates = coordinates_xy_transformed.reshape(-1, 2)

    quadratic_fit_coefficients = np.polyfit(
        flattened_coordinates[:, 0],
        flattened_coordinates[:, 1],
        2
    )

    # Find the minimum distance to the fitted parabola
    displacements_data = [
        _get_distance_to_quadratic(point, quadratic_fit_coefficients)
        for point in flattened_coordinates
    ]

    links_displacements = np.array(
        [ disp for disp, _ in displacements_data ]
    )
    links_fit_projection = np.array(
        [ point for _, point in displacements_data ]
    )

    links_displacements  = links_displacements.reshape(coordinates_xy.shape[0], -1)
    links_fit_projection = links_fit_projection.reshape(coordinates_xy.shape)

    # Compute arc-length of the links trajectory
    links_proj_sorted = np.array(
        [
            links_fit_projection[np.argsort(links_fit_projection[:, link, 0]), link]
            for link in range(links_fit_projection.shape[1])
        ]
    )
    links_proj_sorted = links_proj_sorted.transpose(1,0,2)
    links_proj_diff   = np.diff(links_proj_sorted, axis=0)
    links_proj_arclen = np.sum( np.linalg.norm(links_proj_diff, axis=2), axis=0 )

    plt.plot(coordinates_xy_transformed[:,:,0], coordinates_xy_transformed[:,:,1], '.', c='0.9')
    plt.plot(links_fit_projection[:,:,0], links_fit_projection[:,:,1], '.', c='k')
    plt.plot(links_fit_projection[:,0,0], links_fit_projection[:,0,1], '.', c='r')

    return {
        'links_displacements'        : links_displacements,
        'links_fit_projection'       : links_fit_projection,
        'links_fit_arclen'           : links_proj_arclen,
        'links_positions'            : coordinates_xy,
        'links_positions_transformed': coordinates_xy_transformed,
        'direction_fwd'              : direction_fwd,
        'direction_left'             : direction_left,
        'quadratic_fit_coefficients' : quadratic_fit_coefficients,
    }


with h5py.File(file, 'r') as f:

    link_array = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])

    timestep = np.array(f["timestep"])
    n_iterations = link_array.shape[0]
    it_max = 16500
    times = timestep * np.arange(n_iterations)


    link_pos_xy = link_array[:it_max,:,sc.link_com_position_x:sc.link_com_position_y+1]

    link_displacements = _compute_links_displacements(
        coordinates_xy = link_pos_xy,
    )["links_displacements"]


    plt.figure()
    plt.plot(link_pos_xy[:,0,0],link_pos_xy[:,0,1])
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)


    plt.figure()
    plt.plot(link_displacements)


    plt.show()
