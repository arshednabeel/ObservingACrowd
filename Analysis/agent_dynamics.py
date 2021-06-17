import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from tqdm import tqdm
from typing import Callable, Tuple
from scipy.io import loadmat
from scipy.ndimage import convolve1d
from scipy.spatial import Delaunay
from scipy.stats import binom


f_poly = lambda r: 1 / (r ** 3)
f_exp = lambda r: np.exp(-(r / 3) ** 2)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Palatino']
})

def get_filename(rootdir, nr, deltav, density, realization):
    fname = f'ObservingAndInferring_29April2019_N42_NumberRatio_{nr}_packdens_{density}_delV_{deltav}_Fluc_0_Realization_{realization}.mat'
    fname = os.path.join(rootdir, fname)
    return fname


class AgentDynamics(object):
    
    def __init__(self, filename: str) -> None:
        """ Initialize class using data from MAT file.
        Args:
            filename: Filename of MAT file to load.
        """

        l = filename.split('/')[-1].split('_')
        self.nr, self.density, self.deltav = int(l[4]), float(l[6]), float(l[8])
        mat = loadmat(filename)
        self.pos = np.stack((mat['XA'], mat['YA']), axis=1)
        self.vel = np.stack((mat['VXA'], mat['VYA']), axis=1)
        self.xbounds = [0, float(mat['len'])]
        self.ybounds = [-float(mat['wall']), float(mat['wall'])]
        self.xwidth = self.xbounds[1] - self.xbounds[0]
        self.ywidth = self.ybounds[1] - self.ybounds[0]
        self.agent_radius = mat['R'].squeeze()
        self._wrap_agent_positions()

        self.n = self.pos.shape[0]  # Number of agents
        self.T = self.pos.shape[2]  # Number of timepoints

        self.labels = self.vel[:, 0, 0] > 0
        
    def get_observer_classifications(self, mu: float=0, tau: int=50, f=None) -> np.ndarray:
        """ Get observer predictions, using baseline algorithm, with window size tau.
        Args:
            tau: Window size for prediction
            mu: Scale factor for field-based prediction. If mu=0, velocity-based prediction is used.
        Returns:
            n x T ndarray with predictions for each timepoint.
            Predictions for t < tau are invalid, return array size is kept as n x T for proper alignment.
        """
        if f is None:
            f = lambda r: np.exp(-(r/3) ** 2)
        vel_avg = moving_average(self.vel[:, 0, :], w=tau, axis=1)
        if not mu:
            return vel_avg > 0
        else:
            phi = self.compute_radial_local_fields(f=f)
            phi_avg = moving_average(phi[:, 0, :], w=tau, axis=1)
            return vel_avg > mu * phi_avg

    def compute_distance_based_local_fields(self, f: Callable=None):
        """ 
        Compute distance-based local fields from agent data.
        The local field for an agent is a weighted sum of the velocities of its neighbors (see below).

        Args:
            f: Weighting function that determines how the field decays with distance.
            N: Neighborhood matrix. N[i, j] = 1 if j is a neighbor of i, 0 otherwise.

        TODO Compare performance between this version versus the autograd-friendly version.
        If performance impact is minimal, use the autograd-friendly variant here.
        """

        phi = np.zeros_like(self.vel)
        for t in range(self.T):
            N = get_voronoi_neighbor_matrix(self.pos[:, :, t], periodic=True, xwidth=self.xwidth, ywidth=self.ywidth)
            D = distance_matrix(self.pos[:, :, t], np.array([self.xwidth, self.ywidth]))
           
            if f is None:
                f = self._f_exponential

            F = f(D) * N
            np.fill_diagonal(F, 0)  # Agent does not contribute to its own local field.

            phi[:, :, t] = F @ self.vel[:, :, t]

        return phi

    def compute_radial_local_fields(self, f: Callable=f_exp, scaling_mode='global', sigma_s_global=None):
        """ Compute distance-based local fields from agent data.
        The local field for an agent is a weighted sum of the velocities of its neighbors (see below).

        Args:
            f: Weighting function that determines how the field decays with distance.
        """

        assert scaling_mode in ['global', 'local']

        func_get_field = get_radial_local_field if scaling_mode == 'global' else get_radial_local_field_with_local_scaling

        phi = np.zeros_like(self.vel)
        pos_ = tile_periodic_boundaries(self.pos, self.xwidth, self.ywidth)
        vel_ = np.tile(self.vel, reps=(9, 1, 1))
        for t in range(self.T):
            triangulation = Delaunay(pos_[:, :, t])
            ptr, idx = triangulation.vertex_neighbor_vertices
            for i in range(self.n):               
                neighbors = idx[ptr[i]:ptr[i+1]]
                # neighbors = [n % 9 for n in neighbors]
                phi[i, :, t] = func_get_field(self.pos[i, :, t], 
                    neighbor_pos=pos_[neighbors, :, t],
                    neighbor_vel=vel_[neighbors, :, t], f=f)
                    # neighbor_vel=vel_[neighbors, :, t], f=f, nr=self.nr, sigma_s_global=sigma_s_global)
                
        return phi
            
    def _compute_radial_local_fields_opt(self, f):
        phi = np.zeros_like(self.vel)
        # pos_ = tile_periodic_boundaries(self.pos, self.xwidth, self.ywidth)

        for t in range(self.T):
            N = get_voronoi_neighbor_matrix(self.pos[:, :, t], periodic=True, xwidth=self.xwidth, ywidth=self.ywidth)
            
            # (n, n, 2) array of difference vectors
            diff_vectors = distance_matrix(self.pos[:, :, t], np.array([self.xwidth, self.ywidth]), mode='difference')

            # (n, n) distance matrix
            D = np.linalg.norm(diff_vectors, axis=2)

            # (n, n) array of magnitudes
            radial_vel_mag = np.sum(diff_vectors * self.vel[:, :, t], axis=2) / (D * D)
            np.fill_diagonal(radial_vel_mag, 0)

            # (n, n, 2) array of radial velocities
            radial_vel = (np.expand_dims(radial_vel_mag, axis=2) * diff_vectors)

            F = f(D) * N
            np.fill_diagonal(F, 0)  # Agent does not contribute to its own local field.  

            phi[:, :, t] = np.sum(np.expand_dims(F, axis=2) * radial_vel, axis=1)
                #TODO Confirm this is correct: axis=0 or axis=1 for sum?
        
        return phi

    def get_local_density(self, agent: int, t: int) -> float:
        """ 
        Get the local density around a focal agent at time t, based on distances to its Voronoi neighbors. 
        TODO: If this is needed to be used in batch, the triangulation can be precomputed. Figure out a reasonable implementation if this is required.
        """

        pos_ = tile_periodic_boundaries(self.pos, self.xwidth, self.ywidth)
        triangulation = Delaunay(pos_[:, :, t])
        ptr, idx = triangulation.vertex_neighbor_vertices
        # idx = [i % ad.n for i in idx]  # Convert neighbor indices from tiled to original 0-n indices.
        neighbors = idx[ptr[agent]:ptr[agent+1]]

        diff_vectors = pos_[neighbors, :, t] - pos_[agent, :, t]
        d = np.linalg.norm(diff_vectors, axis=1)
        mean_distance = np.mean(d)

        return 1 / (mean_distance ** 2)

    def get_local_scale_factor(self, agent: int, t: int, f=f_exp, triangulation=None) -> float:
        """
        Compute the analytical scale factor around a focal agent at time t, based on the Voronoi neighborhood.
        """
        
        pos_ = tile_periodic_boundaries(self.pos, self.xwidth, self.ywidth)
        vel_ = np.tile(self.vel, reps=(9, 1, 1))
        if triangulation is None:
            triangulation = Delaunay(pos_[:, :, t])
        ptr, idx = triangulation.vertex_neighbor_vertices
        neighbors = idx[ptr[agent]:ptr[agent+1]]

        diff_vectors = pos_[neighbors, :, t] - pos_[agent, :, t]
        d = np.linalg.norm(diff_vectors, axis=1)
        
        # Length scale: Mean distance
        L_s = np.mean(d)
        f_s = f(L_s)

        # print(f'd={d}')
        # print(f'Ls={L_s}')
        # print(f'fs={f_s}')

        # sigma_s
        # neighbor_vel = vel_[neighbors, :, t]
        # radial_vel_mag = np.sum(neighbor_vel * diff_vectors, axis=1) / (d * d)  # (m, ) array of magnitudes
        # radial_vel = radial_vel_mag * diff_vectors.T   # (m, 2) array of radial velocity vectors.

        # neighbor_vel_sgn = [1 if vel_[neighbors, 0, t] > 0 else -1]
        neighbor_vel_sgn = np.where(vel_[neighbors, 0, t] > 0, 1, -1)
        # print(f"neighbor_vel_sgn.shape: {neighbor_vel_sgn.shape}")
        # print(f"neighbor_vel_sgn: {neighbor_vel_sgn}")
        sigma_s = np.abs(np.sum(neighbor_vel_sgn))
        # print(f'sigma_s: {sigma_s}')

        mu = 1 / (f_s * sigma_s)  # Hack to work around divide-by-zero error.

        return mu

    def visualize(self, t, mu=None, ax=None, face_colors=None, edge_colors=None, focal_agents=None, f=f_exp):
        """ 
        Plot agent positions at a given time point t.
        Args:
            t: Time point for which to visualize
            mu: Scale factor for local field.
            face_colors: Array of face-colors to use.
            edge_colors: Array of edge-colors to use.
            focal_agents: List of focal agents. The agent velocity and local field vectors will be plotted for the focal agent.
                Pass focal_agents='all' to plot for all agents.
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
            phi = mu * self.compute_radial_local_fields(f)
            ax.quiver(self.pos[focal_agents, 0, t],self.pos[focal_agents, 1, t],
                self.vel[focal_agents, 0, t],self.vel[focal_agents, 1, t], color=[.5, .5, 1], width=0.005)
            ax.quiver(self.pos[focal_agents, 0, t],self.pos[focal_agents, 1, t],
                phi[focal_agents, 0, t],phi[focal_agents, 1, t], color=[1, .5, 0], width=0.005)

        if show_plot:
            plt.show()
        else:
            return ax

    def _wrap_agent_positions(self):
        def _wrap(A, lb, ub):
            return ((A - lb) % (ub - lb)) + lb

        self.pos[:, 0, :] = _wrap(self.pos[:, 0, :], self.xbounds[0], self.xbounds[1])
        self.pos[:, 1, :] = _wrap(self.pos[:, 1, :], self.ybounds[0], self.ybounds[1])

    def _f_exponential(r: float, epsilon: float=3):
        return np.exp(-(r/epsilon) ** 2)

    def _f_polynomial(r: float, beta: -3):
        return r^beta

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
        mode: If mode is 'distance', return the distance matrix. If mode is 'difference', return the (N, N, d) array of differences.

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
        xwidth, ywidth: Width along the x and y dimensions respectively
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

def get_distance_based_local_field(pos: Tuple[np.float64, np.float64], 
        neighbor_pos: np.ndarray, neighbor_vel: np.ndarray,
        f: Callable) -> Tuple:
    """
    Similar to AgentDynamics.get_radial_local_field, implemented in an autograd-friendly way.
    Returns the local field value at (x, y), computed using neighbor positions and velocities given in pos and vel arrays.

    Args:
        pos: Coordinates for the point to compute local field for (coordinates of the focal agent)
        neighbor_pos: (m, 2) array of position coordinates of neighbors.
        neighbor_vel: (m, 2) array of velocity coordinates of neighbors.
        f: Function that determines how field strength decays with distance.

    Returns:
        phi, the local field evaluated at x, y.
    """

    # pos = np.broadcast_to(pos, neighbor_pos.shape)
    pos = np.array(pos)
    d = np.linalg.norm(neighbor_pos - pos, axis=1)  # (m, ) array of distances
    phi = np.sum(f(d) * neighbor_vel, axis=1)

    return phi

def get_radial_local_field_with_local_scaling(pos: Tuple[np.float64, np.float64], 
        neighbor_pos: np.ndarray, neighbor_vel: np.ndarray,
        f: Callable, nr, sigma_s_global=None, sigma_mode='global') -> Tuple:
    """
    Returns the local field value at (x, y), computed using neighbor positions and velocities given in pos and vel arrays.

    Args:
        pos: Coordinates for the point to compute local field for (coordinates of the focal agent)
        neighbor_pos: (m, 2) array of position coordinates of neighbors.
        neighbor_vel: (m, 2) array of velocity coordinates of neighbors.
        f: Function that determines how field strength decays with distance.

    Returns:
        phi, the local field evaluated at x, y.
    """

    # Step 1: Compute phi
    diff_vectors = neighbor_pos - pos   # (m, 2) array of difference vectors
    d = np.linalg.norm(diff_vectors, axis=1)  # (m, ) array of distances
    radial_vel_mag = np.sum(neighbor_vel * diff_vectors, axis=1) / (d * d)  # (m, ) array of magnitudes
        # row-wise dot-product, normalized by d

    radial_vel = radial_vel_mag * diff_vectors.T   # (m, 2) array of radial velocity vectors.
    phi = np.sum(f(d) * radial_vel, axis=1)

    # Step 2: Compute mu (local scaling factor)    
    # Length scale: Mean distance
    L_s = np.mean(d)
    f_s = f(L_s)
    if sigma_mode == 'local':
        neighbor_vel_sgn = np.where(neighbor_vel[:, 0] > 0, 1, -1)
        sigma_s = np.abs(np.sum(neighbor_vel_sgn))
    if sigma_mode == 'global':  
        sigma_s = sigma_s_global

    mu = 1 / (f_s * sigma_s) 

    return mu * phi

def get_radial_local_field(pos: Tuple[np.float64, np.float64], 
        neighbor_pos: np.ndarray, neighbor_vel: np.ndarray,
        f: Callable) -> Tuple:
    """
    Returns the local field value at (x, y), computed using neighbor positions and velocities given in pos and vel arrays.

    Args:
        pos: Coordinates for the point to compute local field for (coordinates of the focal agent)
        neighbor_pos: (m, 2) array of position coordinates of neighbors.
        neighbor_vel: (m, 2) array of velocity coordinates of neighbors.
        f: Function that determines how field strength decays with distance.

    Returns:
        phi, the local field evaluated at x, y.
    """

    diff_vectors = neighbor_pos - pos   # (m, 2) array of difference vectors
    d = np.linalg.norm(diff_vectors, axis=1)  # (m, ) array of distances
    radial_vel_mag = np.sum(neighbor_vel * diff_vectors, axis=1) / (d * d)  # (m, ) array of magnitudes
        # row-wise dot-product, normalized by d

    radial_vel = radial_vel_mag * diff_vectors.T   # (m, 2) array of radial velocity vectors.
    phi = np.sum(f(d) * radial_vel, axis=1)

    # print(f'phi.shape = {phi.shape}')
    return phi


def get_voronoi_neighbor_matrix(pos: np.ndarray, periodic=True, xwidth: float=None, ywidth: float=None):
    """ Get Voronoi neighbours of each agent in matrix form, with periodic boundary conditions. 
    Args:
        pos: n x 2 array of agent positions.
        periodic: Whether to use periodic boundary conditions.
    Returns:
        N: n x n array such that N[i, j] = 1 if agent j is a neighbor of agent i, 0 otherwise. 
    """
    
    assert periodic == False or (xwidth is not None and ywidth is not None), \
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

    

