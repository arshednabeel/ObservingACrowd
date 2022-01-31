import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from typing import Callable
from scipy.io import loadmat
from scipy.ndimage import convolve1d
from scipy.spatial import Delaunay


def f_poly(r): return 1 / (r ** 3)
def f_exp(r): return np.exp(-(r / 3) ** 2)


def get_filename(rootdir, nr, deltav, density, realization):
    fname = f'ObservingAndInferring_29April2019_N42_' \
        f'NumberRatio_{nr}_packdens_{density}_delV_{deltav}_Fluc_0_Realization_{realization}.mat'
    fname = os.path.join(rootdir, fname)
    return fname


class AgentDynamics(object):
    
    def __init__(self, filename: str, f: Callable = f_exp) -> None:
        """ Initialize class using data from MAT file.
        Args:
            filename: Filename of MAT file to load.
            f: (Distance based) weighting function
        """

        chop = filename.split('/')[-1].split('_')
        self.nr, self.density, self.deltav = int(chop[4]), float(chop[6]), float(chop[8])
        mat = loadmat(filename)
        self.pos = np.stack((mat['XA'], mat['YA']), axis=1)    # Shape: (n, 2, T)
        self.vel = np.stack((mat['VXA'], mat['VYA']), axis=1)  # Shape: (n, 2, T)
        self.xbounds = [0, float(mat['len'])]
        self.ybounds = [-float(mat['wall']), float(mat['wall'])]
        self.xwidth = self.xbounds[1] - self.xbounds[0]
        self.ywidth = self.ybounds[1] - self.ybounds[0]
        self.agent_radius = mat['R'].squeeze()
        self._wrap_agent_positions()

        self.n = self.pos.shape[0]  # Number of agents
        self.T = self.pos.shape[2]  # Number of timepoints

        self.f = f  # (Distance-based) weighting function

        self.labels = self.vel[:, 0, 0] > 0   # Shape: (n, )

    def compute_local_fields(self, voronoi=False, radial=False, relative=False):
        """ Local field (neighbourhood parameter computation function. Derived classes should implement this function
        with appropriate method of computation. """

        phi = np.zeros_like(self.vel)

        for t in range(self.T):

            # (n, n, 2) array of difference vectors
            diff_vectors = distance_matrix(self.pos[:, :, t], np.array([self.xwidth, self.ywidth]), mode='difference')

            # (n, n) distance matrix
            D = np.linalg.norm(diff_vectors, axis=2)
            if relative:
                v_j = np.empty((self.n, self.n, 2))  # Shape (2, n, n)
                v_j[:, :, 0] = np.subtract.outer(self.vel[:, 0, t], self.vel[:, 0, t]).T
                v_j[:, :, 1] = np.subtract.outer(self.vel[:, 1, t], self.vel[:, 1, t]).T
            else:
                v_j = self.vel[:, :, t]

            if radial:
                # (n, n) array of magnitudes
                radial_vel_mag = np.sum(diff_vectors * v_j, axis=2) / (D * D)
                np.fill_diagonal(radial_vel_mag, 0)

                # (n, n, 2) array of radial velocities
                v_j = (np.expand_dims(radial_vel_mag, axis=2) * diff_vectors)

            if voronoi:
                N = get_voronoi_neighbor_matrix(self.pos[:, :, t], periodic=True, xwidth=self.xwidth, ywidth=self.ywidth)
                F = self.f(D) * N
            else:
                F = self.f(D)
            np.fill_diagonal(F, 0)  # Agent does not contribute to its own local field.

            phi[:, :, t] = np.sum(np.expand_dims(F, axis=2) * v_j, axis=1)

        return phi

    def get_observer_classifications(self, mu: float = 0, tau: int = 50) -> np.ndarray:
        """ Get observer predictions, using baseline algorithm, with window size tau.
        Args:
            tau: Window size for prediction
            mu: Scale factor for field-based prediction. If mu=0, velocity-based prediction is used.
        Returns:
            n x T ndarray with predictions for each timepoint.
            Predictions for t < tau are invalid, return array size is kept as n x T for proper alignment.
        """

        vel_avg = moving_average(self.vel[:, 0, :].squeeze(), w=tau, axis=1)
        if not mu:
            return vel_avg > 0
        else:
            phi = self.compute_local_fields()
            phi_avg = moving_average(phi[:, 0, :].squeeze(), w=tau, axis=1)
            return vel_avg > mu * phi_avg

    def visualize(self, t, mu=None, ax=None, face_colors=None, edge_colors=None, focal_agents=None):
        """ 
        Plot agent positions at a given time point t.
        Args:
            t: Time point for which to visualize
            mu: Scale factor for local field.
            face_colors: Array of face-colors to use.
            edge_colors: Array of edge-colors to use.
            focal_agents: List of focal agents. The agent velocity and local field vectors will be plotted
                for the focal agent. Pass focal_agents='all' to plot for all agents.
            ax: (Optional) axis to plot to. If not given, a new figure will be created.
        """

        if focal_agents:
            assert mu is not None, 'To highlight focal agents, provide mu (scale factor) argument.'

        show_plot = False
        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_axes([0, 0, 1, 1], frameon=False)
            ax.set(xlim=self.xbounds, ylim=self.ybounds)
            show_plot = True

        ax.clear()

        if focal_agents == 'all':
            focal_agents = range(self.n)

        agents = [Circle(xy=(self.pos[i, 0, t], self.pos[i, 1, t]), radius=self.agent_radius) for i in range(self.n)]
        p = PatchCollection(agents)
        p.set_edgecolor(edge_colors)
        p.set_facecolor(face_colors)
        p.set_linewidth(2)
        ax.add_collection(p)

        if focal_agents:
            phi = mu * self.compute_local_fields()
            ax.quiver(self.pos[focal_agents, 0, t], self.pos[focal_agents, 1, t],
                      self.vel[focal_agents, 0, t], self.vel[focal_agents, 1, t], color=[.5, .5, 1], width=0.005)
            ax.quiver(self.pos[focal_agents, 0, t], self.pos[focal_agents, 1, t],
                      phi[focal_agents, 0, t], phi[focal_agents, 1, t], color=[1, .5, 0], width=0.005)

        if show_plot:
            plt.show()
        else:
            return ax

    def _wrap_agent_positions(self):
        def _wrap(A, lb, ub):
            return ((A - lb) % (ub - lb)) + lb

        self.pos[:, 0, :] = _wrap(self.pos[:, 0, :], self.xbounds[0], self.xbounds[1])
        self.pos[:, 1, :] = _wrap(self.pos[:, 1, :], self.ybounds[0], self.ybounds[1])




# ------------- Helper Functions ----------------

def moving_average(X, w, axis=0):
    """ Causal moving average computed along a specified axis.
    out[t] is the average of X[t - w] ... X[t]. 
    """
    
    origin = -(w // 2)
    return convolve1d(X, np.ones(w), axis=axis, mode='nearest', origin=origin) / w


def distance_matrix(rs, L, mode='distance'):
    """
    Returns the distance matrix between all pairs of points in rs, given a periodic box of size L.

    Args:
        rs : (N, d) array_like; input vectors
        L : (d, ) array_like or scalar; size of the box
        mode: If mode is 'distance', return the distance matrix. If mode is 'difference',
              return the (N, N, d) array of differences.

    Returns:
        (N, N) ndarray of pairwise distances or (N, N, d) array of pairwise differences
    """

    assert mode in ['distance', 'difference']

    diffs = np.array([np.subtract.outer(rd, rd) for rd in rs.T]).T
    # dist_vectors[i,j,:] is the difference vector between rs[i,:] and rs[j,:]
    dist_vectors = np.remainder(diffs + L/2., L) - L/2.
    if mode == 'distance':
        return np.linalg.norm(dist_vectors, axis=2)
    else:
        return dist_vectors


def tile_periodic_boundaries(pos: np.ndarray, xwidth: float, ywidth: float):
    """
    Tile (repeat) given points (agents) on a 3x3 grid.
    Args:
        pos: (n, 2, T) array of agent position coordinates.
        xwidth:
        ywidth: Width along the x and y dimensions respectively
    Returns:
        (9*n, 2, T) augmented arrays of position coordinates.
    """

    if len(pos.shape) == 2:
        pos = np.expand_dims(pos, axis=2)
    n = pos.shape[0]
    pos = np.tile(pos, reps=(9, 1, 1))
    # Center Tile: XAaug[:n, :], YAaug[:n, :] -- unchanged

    # Top Tile: XAaug is unchanged
    pos[n:2 * n, 1, :] += ywidth

    # Bottom Tile: XAaug is unchanged
    pos[2 * n:3 * n, 1, :] -= ywidth

    # Right Tile: YAaug is unchanged
    pos[3 * n:4 * n, 0, :] += xwidth

    # Left Tile: YAaug is unchanged
    pos[4 * n:5 * n, 0, :] -= xwidth

    # Top Right Tile
    pos[5 * n:6 * n, 0, :] += xwidth
    pos[5 * n:6 * n, 1, :] += ywidth

    # Top Left Tile
    pos[6 * n:7 * n, 0, :] -= xwidth
    pos[6 * n:7 * n, 1, :] += ywidth

    # Bottom Right Tile
    pos[7 * n:8 * n, 0, :] += xwidth
    pos[7 * n:8 * n, 1, :] -= ywidth

    # Bottom Left Tile
    pos[8 * n:, 0, :] -= xwidth
    pos[8 * n:, 1, :] -= ywidth

    return pos


def get_voronoi_neighbor_matrix(pos: np.ndarray, periodic=True, xwidth: float = None, ywidth: float = None):
    """ Get Voronoi neighbours of each agent in matrix form, with periodic boundary conditions. 
    Args:
        pos: n x 2 array of agent positions.
        periodic: Whether to use periodic boundary conditions.
        xwidth:
        ywidth: Width along the x and y dimensions respectively
    Returns:
        N: n x n array such that N[i, j] = 1 if agent j is a neighbor of agent i, 0 otherwise. 
    """
    
    assert not periodic or (xwidth is not None and ywidth is not None), \
        'If periodic, specify xwidth and ywidth'
    
    n_ = pos.shape[0]
    
    if periodic:
        pos = tile_periodic_boundaries(pos, xwidth, ywidth).squeeze()
    
    n = pos.shape[0]
    # print(f'pos.shape: {pos.shape}')
    ptr, idx = Delaunay(pos).vertex_neighbor_vertices
    # print(ptr)
    N = np.zeros((n, n))
    
    for i in range(n):
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.vertex_neighbor_vertices.html
        for neighbor in idx[ptr[i]:ptr[i+1]]:
            N[i, neighbor] = 1
    
    if periodic:
        # Fold N over itself to get n x n neighborhood matrix
        N_ = N[:n_, :n_].copy()
        for ii in range(1, 9):
            # N_ += N[ii * n_:(ii + 1) * n_, :n_] + N[:n_, ii * n_:(ii + 1) * n_]
            N_ += N[:n_, ii * n_:(ii + 1) * n_]
        
        N = N_
    
    return N
