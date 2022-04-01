import os, sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from celluloid import Camera
from tqdm import tqdm, trange
from scipy.spatial import Delaunay

from agent_dynamics import ADFieldBased as AgentDynamics
from agent_dynamics import tile_periodic_boundaries
from classify import DataClassifier, moving_average, f_exp, f_poly, get_scale_factor_analytical

# --- GLOBAL CONSTANTS ---
C_RED = (178/255, 24/255, 43/255)
C_BLUE = (33/255, 102/255, 172/255)

C_RED_DARK = (103/255, 0/255, 31/255)
C_BLUE_DARK = (5/255, 48/255, 97/255)

C_RED_LIGHT = (231/255, 72/255, 94/255)
C_BLUE_LIGHT = (101/255, 163/255, 226/255)

rootdir = '/Users/nabeel/Data/ObservingAndInferring/SimData'
cachedir = '/Volumes/Backyard/Data/cache/expkernel'
outdir = '/Users/nabeel/Data/output/'
summaryfile = '/Users/nabeel/Data/output/classification_metrics_expkernel.npz'
nmfile = '/Users/nabeel/Data/output/misclassifications.npz'


# --- ANIMATION FUNCTIONS ---
def animate_basic(nr, density, deltav, realization=None, timespan=None, outfile=None):
    if timespan is None:
        timespan = range(1001)

    filename = f'ObservingAndInferring_29April2019_' \
               f'N42_NumberRatio_{nr}_packdens_{density}_delV_{deltav}_Fluc_0_Realization_{realization}.mat'
    filename = os.path.join(rootdir, filename)
    ad = AgentDynamics(filename)

    pos = tile_periodic_boundaries(ad.pos, ad.xwidth, ad.ywidth)
    labels = np.tile(ad.labels, reps=(9,))

    # face_colors = np.ones((ad.n * 9, 3))
    face_colors = np.zeros((ad.n * 9, 3))
    edge_colors = np.zeros((ad.n * 9, 3))

    # face_colors[labels, :] = C_BLUE
    # face_colors[~labels, :] = C_RED
    edge_colors[labels, :] = C_BLUE_DARK
    edge_colors[~labels, :] = C_RED_DARK

    fig = plt.figure(figsize=(7, 7), facecolor='k')
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set(xlim=ad.xbounds, ylim=ad.ybounds, facecolor='k')
    cam = Camera(fig)

    v_mix = ad.vel.mean(axis=0)
    for t in timespan:
        agents = [Circle(xy=(pos[i, 0, t], pos[i, 1, t]), radius=ad.agent_radius)
                  for i in range(ad.n * 9)]
        p = PatchCollection(agents)
        p.set_edgecolor(edge_colors)
        p.set_facecolor(face_colors)
        p.set_linewidth(6)
        ax.add_collection(p)

        for i in range(ad.n):
            ax.text(ad.pos[i, 0, t], ad.pos[i, 1, t], f'{i}', color=[1, 1, 1])

        cam.snap()

    animation = cam.animate()
    if outfile:
        animation.save(outfile, fps=30)
        plt.close(fig)
    else:
        plt.show()


def animate_velocity_traces(filename, outfile=None, timespan=None):
    """ Animate a horizontal line moving across a velocity trace (as in Figure 1)"""
    pass


def animate_with_vectors(nr, density, deltav, realization, outfile, focal_agents=None, timespan=None, quiverscale=1.,
                         label_mode='baseline'):
    """ Animate dynamics from a given MAT file, highlighting the specified list of focal agents. """

    assert label_mode in ['baseline', 'field']

    filename = f'ObservingAndInferring_29April2019_' \
               f'N42_NumberRatio_{nr}_packdens_{density}_delV_{deltav}_Fluc_0_Realization_{realization}.mat'
    filename = os.path.join(rootdir, filename)

    mu = get_scale_factor_analytical(nr=nr, deltav=deltav, density=density)

    ad = AgentDynamics(filename)
    if label_mode == 'baseline':
        preds = ad.get_observer_classifications(mu=0, voronoi=True, radial=True, relative=False, mean=True, cos=False)
    else:
        preds = ad.get_observer_classifications(mu=mu, voronoi=True, radial=True, relative=False, mean=True, cos=False)
    phi = mu * ad.compute_local_fields(voronoi=True, radial=True, relative=False, mean=True, cos=False)
    vel_avg = moving_average(ad.vel, w=50, axis=2)
    phi_avg = moving_average(phi, w=50, axis=2)

    fig = plt.figure(figsize=(7, 7), facecolor='k')
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set(xlim=ad.xbounds, ylim=ad.ybounds)
    ax.set_facecolor('k')

    if focal_agents == 'all':
        focal_agents = range(ad.n)

    if timespan is None:
        timespan = range(ad.T)

    face_colors = np.zeros((ad.n, 3))
    edge_colors = 0.4 * np.ones((ad.n, 3))

    for agent in focal_agents:
        edge_colors[agent, :] = C_BLUE_DARK if ad.labels[agent] else C_RED_DARK

    cam = Camera(fig)
    for t in timespan:
        neighbors = np.zeros(ad.n, dtype=bool)
        neighbors[focal_agents] = True
        pos_ = tile_periodic_boundaries(ad.pos, ad.xwidth, ad.ywidth)
        triangulation = Delaunay(pos_[:, :, t])  # Compute Delaunay triangulation on a tiled version.
        ptr, idx = triangulation.vertex_neighbor_vertices
        idx = [i % ad.n for i in idx]  # Convert neighbor indices to original 0-n indices.
        for agent in focal_agents:
            face_colors[agent] = C_BLUE if preds[agent, t] else C_RED
            neighbors[idx[ptr[agent]:ptr[agent + 1]]] = True

        # ax.clear()
        agents = [Circle(xy=(ad.pos[i, 0, t], ad.pos[i, 1, t]), radius=ad.agent_radius) for i in range(ad.n)]
        p = PatchCollection(agents)
        p.set_edgecolor(edge_colors)
        p.set_facecolor(face_colors)
        p.set_linewidth(6)
        ax.add_collection(p)
        if label_mode == 'field':
            ax.quiver(ad.pos[neighbors, 0, t], ad.pos[neighbors, 1, t],
                      vel_avg[neighbors, 0, t], vel_avg[neighbors, 1, t],
                      scale=quiverscale, scale_units='inches', color=[.5, .5, 1], width=0.005)
            ax.quiver(ad.pos[focal_agents, 0, t], ad.pos[focal_agents, 1, t],
                      phi_avg[focal_agents, 0, t], phi_avg[focal_agents, 1, t],
                      scale=quiverscale, scale_units='inches', color=[1, .5, 0], width=0.005)
        else:
            ax.quiver(ad.pos[focal_agents, 0, t], ad.pos[focal_agents, 1, t],
                      vel_avg[focal_agents, 0, t], vel_avg[focal_agents, 1, t],
                      scale=quiverscale, scale_units='inches', color=[.5, .5, 1], width=0.005)

        # plt.show()
        cam.snap()

        # edge_colors = np.zeros((ad.n, 3))

    animation = cam.animate()
    if outfile:
        animation.save(outfile, fps=30)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    # outdir = '/Users/nabeel/Documents/Research/Writing/Observing and Inferring a Collective/ChemE IISc Symposium/'
    outdir = '/Users/nabeel/Data/output/movies'
    os.makedirs(outdir, exist_ok=True)

    density = 0.45792
    deltav = 1

    nr = 4
    outfile = os.path.join(outdir, f'WithLabels_nr_{nr}_dens_{density}_deltav_{deltav}.mp4')
    # animate_with_vectors(nr=nr, deltav=deltav, density=density, realization=98,
    #                      focal_agents=[39, ], timespan=None, label_mode='field',
    #                      outfile=outfile, quiverscale=1)
    animate_basic(nr=nr, density=density, deltav=deltav, realization=1, outfile=outfile)


    # nr = 9
    # outfile = os.path.join(outdir, f'Movie_nr_{nr}_dens_{density}_deltav_{deltav}_simple.mp4')
    # # animate_with_vectors(nr=nr, deltav=deltav, density=density, realization=1,
    # #                      focal_agents=[9, ], timespan=None, label_mode='field',
    # #                      outfile=outfile, quiverscale=1)
    # animate_basic(nr=nr, density=density, deltav=deltav, realization=1, outfile=outfile)
    #
    # nr = 21
    # outfile = os.path.join(outdir, f'Movie_nr_{nr}_dens_{density}_deltav_{deltav}_simple.mp4')
    # # animate_with_vectors(nr=nr, deltav=deltav, density=density, realization=1,
    # #                      focal_agents=[0, ], timespan=None, label_mode='field',
    # #                      outfile=outfile, quiverscale=1)
    # animate_basic(nr=nr, density=density, deltav=deltav, realization=1, outfile=outfile)
    # for r in np.random.randint(0, 42, 5):
    #     outfile = os.path.join(outdir, f'M1_{r}.mp4')
    #     animate_basic(nr=4, density=0.45792, deltav=1, realization=r, outfile=outfile)
    # outfile = os.path.join(outdir, 'M1_2.mp4')
    # animate_basic(nr=21, density=0.45792, deltav=1, realization=63, outfile=outfile)
    # outfile = os.path.join(outdir, 'M1_3.mp4')
    # animate_basic(nr=9, density=0.57706, deltav=0.5, realization=5, outfile=outfile)
    # outfile = os.path.join(outdir, 'M2_2.mp4')
    # animate_basic(nr=9, density=0.57706, deltav=1.5, realization=5, outfile=outfile)
    # outfile = os.path.join(outdir, 'M3_1.mp4')
    # outfile = os.path.join(outdir, 'M3_2.mp4')
    # animate_with_vectors(nr=21, deltav=0.5, density=0.45792, realization=69,
    #                      focal_agents=[14, ], timespan=range(200, 650),
    #                      outfile=outfile, quiverscale=0.6)
