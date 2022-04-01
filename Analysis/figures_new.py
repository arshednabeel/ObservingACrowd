import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from joblib import Parallel, delayed
from agent_dynamics import ADFieldBased as AgentDynamics, moving_average, tile_periodic_boundaries
from classify import DataClassifier, get_scale_factor_analytical

from tqdm import tqdm

RED, BLUE = 0, 1
N = 42

plt.rcParams.update(
    {
        'font.family': 'Palatino',
        'font.size': 28, #32,
    }
)

f = lambda r: 1

# Colors
C_RED = (178/255, 24/255, 43/255)
C_BLUE = (33/255, 102/255, 172/255)

C_RED_DARK = (103/255, 0/255, 31/255)
C_BLUE_DARK = (5/255, 48/255, 97/255)

# summaryfile = '/Users/nabeel/Data/output/140222_noproj.npz'
# summaryfile = '/Users/nabeel/Data/output/constkernel_vor1_rel0_rad0_mean1.npz'


def plot_nr_vs_misclassifications(density, deltav, both=True, title=None, outfile=None, ax=None, show_labels=True):

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        showplot = True
    else:
        showplot = False

    data = np.load(summaryfile)
    nrs, deltavs, densities = data['nrs'], data['deltavs'], data['densities']
    nrs = np.array([nr / N for nr in nrs])
    jdeltav = np.argwhere(deltavs == deltav)
    kdensity = np.argwhere(densities == density)

    # baseline_cms = data['baseline_cms'] / (1001 * 100)
    baseline_cms = data['field_cms_analytical'] / (1001 * 100)
    field_cms = data['field_cms_svm_unbalanced'] / (1001 * 100)
    # field_cms = data['field_cms_analytical'] / (1001 * 100)

    if both:
        alpha = 0.6
        marker = 'o--'
    else:
        alpha = 1
        marker = 'o-'

    ax.plot(nrs,
            baseline_cms[:, jdeltav, kdensity, RED, BLUE].squeeze() / (N * nrs),
            marker, color=C_RED, alpha=alpha,
            label='C-I (Smaller Group)',
            linewidth=3, markersize=10)
    ax.plot(nrs,
            baseline_cms[:, jdeltav, kdensity, BLUE, RED].squeeze() / (N * (1 - nrs)),
            marker, color=C_BLUE, alpha=alpha,
            label='C-I (Larger Group)',
            linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=2)

    if both:
        ax.plot(nrs,
                field_cms[:, jdeltav, kdensity, RED, BLUE].squeeze() / (N * nrs),
                's-', color=C_RED_DARK,
                label='C-II (Smaller Group)',
                linewidth=3, markersize=10)
        ax.plot(nrs,
                field_cms[:, jdeltav, kdensity, BLUE, RED].squeeze() / (N * (1 - nrs)),
                's-', color=C_BLUE_DARK,
                label='C-II (Larger Group)',
                linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=2)
    if show_labels:
        ax.set(xlabel='$N_r$', ylabel='$p_m$')#, ylim=[-0.05, 1.05])

    if title:
        ax.set(title=title)
    if showplot:
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()


def plot_nr_vs_total_misclassifications(density, deltav, mode='unbalanced', both=True, title=None, outfile=None):
    assert mode in ['unbalanced', 'balanced']

    fig, ax = plt.subplots(figsize=(8, 6.5))

    data = np.load(summaryfile)
    nrs, deltavs, densities = data['nrs'], data['deltavs'], data['densities']
    nrs = np.array([nr / N for nr in nrs])
    jdeltav = np.argwhere(deltavs == deltav)
    kdensity = np.argwhere(densities == density)

    baseline_cms = data['baseline_cms'] / (1001 * 100)
    # baseline_cms = data['mixvel_cms'] / (1001 * 100)
    # field_cms = data['field_cms_svm_unbalanced'] / (1001 * 100)
    field_cms = data['field_cms_analytical'] / (1001 * 100)

    if mode == 'unbalanced':
        tot_baseline = (baseline_cms[:, jdeltav, kdensity, RED, BLUE].squeeze() +
                        baseline_cms[:, jdeltav, kdensity, BLUE, RED].squeeze()) / N
    else:
        tot_baseline = (baseline_cms[:, jdeltav, kdensity, RED, BLUE].squeeze() / (N * nrs) +
                        baseline_cms[:, jdeltav, kdensity, BLUE, RED].squeeze() / (N * (1 - nrs))) / 2

    ax.plot(nrs,
            tot_baseline,
            'o--', color=C_RED,
            label='C-I (Smaller Group)',
            linewidth=3, markersize=10)

    if both:
        if mode == 'unbalanced':
            tot_field = (field_cms[:, jdeltav, kdensity, RED, BLUE].squeeze() +
                         field_cms[:, jdeltav, kdensity, BLUE, RED].squeeze()) / N
        else:
            tot_field = (field_cms[:, jdeltav, kdensity, RED, BLUE].squeeze() / (N * nrs) +
                         field_cms[:, jdeltav, kdensity, BLUE, RED].squeeze() / (N * (1 - nrs))) / 2

        ax.plot(nrs,
                tot_field,
                's-', color=C_RED_DARK,
                label='C-II (Smaller Group)',
                linewidth=3, markersize=10)

    ax.set(xlabel='$N_r$', ylabel='$p_m$')#, ylim=[-0.05, 1.05])

    if title:
        ax.set(title=title)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


def plot_actual_vs_estimated_nr(density, deltav, both=True, title=None, outfile=None, ax=None, show_labels=True):
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        showplot = True
    else:
        showplot = False

    data = np.load(summaryfile)
    nrs, deltavs, densities = data['nrs'], data['deltavs'], data['densities']
    nrs = np.array([nr / N for nr in nrs])
    jdeltav = np.argwhere(deltavs == deltav)
    kdensity = np.argwhere(densities == density)

    # baseline_cms = data['baseline_cms'] / (1001 * 100)
    baseline_cms = data['field_cms_analytical'] / (1001 * 100)
    # baseline_cms = data['mixvel_cms'] / (1001 * 100)
    field_cms = data['field_cms_svm_unbalanced'] / (1001 * 100)
    # field_cms = data['field_cms_analytical'] / (1001 * 100)

    nr_est_baseline = (baseline_cms[:, jdeltav, kdensity, RED, RED] +
                       baseline_cms[:, jdeltav, kdensity, BLUE, RED]).squeeze() / N
    nr_est_field = (field_cms[:, jdeltav, kdensity, RED, RED] +
                    field_cms[:, jdeltav, kdensity, BLUE, RED]).squeeze() / N

    if both:
        alpha = 0.6
        marker = 'o--'
    else:
        alpha = 1
        marker = 'o-'

    ax.plot(nrs, nr_est_baseline,
            marker, color=C_RED, alpha=alpha,
            label='Simple Observer',
            linewidth=3, markersize=10)

    if both:
        ax.plot(nrs, nr_est_field,
                'o-', color=C_RED_DARK,
                label='Simple Observer',
                linewidth=3, markersize=10)

    ax.axline(xy1=(0, 0), slope=1, linewidth=2, color=(0.6, 0.6, 0.6))
    if show_labels:
        ax.set(xlabel='$N_r$ (Actual)', ylabel='$\\hat N_r$ (Estimated)')#, ylim=[-0.05, 1.05])
    ax.axis('square')
    # ax.grid(True)

    if title:
        ax.set(title=title)

    if showplot:
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()


def pm_plots_multipanel(densities, deltavs, both=True):
    assert len(densities) == 3
    assert len(deltavs) == 3

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    plot_nr_vs_misclassifications(ax=ax1, density=densities[0], deltav=deltavs[0], both=both, show_labels=True)
    plot_nr_vs_misclassifications(ax=ax2, density=densities[1], deltav=deltavs[1], both=both, show_labels=False)
    plot_nr_vs_misclassifications(ax=ax3, density=densities[2], deltav=deltavs[2], both=both, show_labels=False)

    plt.tight_layout()
    plt.show()


def nr_plots_multipanel(densities, deltavs, both=True):
    assert len(densities) == 3
    assert len(deltavs) == 3

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    plot_actual_vs_estimated_nr(ax=ax1, density=densities[0], deltav=deltavs[0], both=both, show_labels=True)
    plot_actual_vs_estimated_nr(ax=ax2, density=densities[1], deltav=deltavs[1], both=both, show_labels=False)
    plot_actual_vs_estimated_nr(ax=ax3, density=densities[2], deltav=deltavs[2], both=both, show_labels=False)

    plt.tight_layout()
    plt.show()


def batch_plot_kdes(nrs, densities, deltavs):

    def plot_func(nr, density, deltav):
        filename = os.path.join(outdir, f'kde_nr_{nr}_density_{density}_deltav_{deltav}.png')
        dc = DataClassifier(rootdir=rootdir, nr=nr, density=density, deltav=deltav,
                            f=f, realizations=100, cachedir=None)
        dc.show_distributions_2d(scale_factor=dc.scale_factor_analytical, color1=C_BLUE_DARK, color2=C_RED_DARK,
                                 filename=filename)

    Parallel(n_jobs=4)(delayed(plot_func)(nr, density, deltav)
                       for nr in nrs
                       for density, deltav in zip(densities, deltavs)
                       )


def show_agent_snapshot(nr, density, deltav, focal_agents=(), realization=None, t=None, show_agent_ids=False):
    if realization is None:
        realization = np.random.randint(1, 101)
    if t is None:
        t = np.random.randint(100, 1001)

    filename = f'ObservingAndInferring_29April2019_' \
               f'N42_NumberRatio_{nr}_packdens_{density}_delV_{deltav}_Fluc_0_Realization_{realization}.mat'
    filename = os.path.join(rootdir, filename)
    ad = AgentDynamics(filename)
    pos = tile_periodic_boundaries(ad.pos, ad.xwidth, ad.ywidth)
    if not focal_agents:
        focal_agents = range(ad.n)

    face_colors = np.ones((ad.n, 3))
    edge_colors = np.ones((ad.n, 3))
    for agent in focal_agents:
        face_colors[agent, :] = C_BLUE if ad.labels[agent] else C_RED
    # face_colors[ad.labels, :] = C_BLUE
    # face_colors[~ad.labels, :] = C_RED

    edge_colors[ad.labels, :] = C_BLUE_DARK
    edge_colors[~ad.labels, :] = C_RED_DARK

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set(xlim=ad.xbounds, ylim=ad.ybounds)

    agents = [Circle(xy=(pos[i, 0, t], pos[i, 1, t]), radius=ad.agent_radius) for i in range(ad.n * 9)]
    p = PatchCollection(agents)
    p.set_edgecolor(edge_colors)
    p.set_facecolor(face_colors)
    p.set_linewidth(6)
    ax.add_collection(p)

    if show_agent_ids:
        for i in range(ad.n):
            ax.text(ad.pos[i, 0, t], ad.pos[i, 1, t], f'{i}')

    plt.show()


def plot_velocity_timeseries(nr, deltav, density, realization, agent, timespan):
    filename = f'ObservingAndInferring_29April2019_' \
               f'N42_NumberRatio_{nr}_packdens_{density}_delV_{deltav}_Fluc_0_Realization_{realization}.mat'
    filename = os.path.join(rootdir, filename)

    mu = get_scale_factor_analytical(nr=nr, deltav=deltav, density=density)

    ad = AgentDynamics(filename)

    phi = mu * ad.compute_local_fields(voronoi=True, radial=True, relative=False, mean=True, cos=False)

    v = moving_average(ad.vel, w=10, axis=2)[agent, 0, timespan]
    phi = moving_average(phi, w=10, axis=2)[agent, 0, timespan]
    # v = ad.vel[agent, 0, timespan]
    # phi = phi[agent, 0, timespan]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axhline(0, ls=':', color=(0.3, 0.3, 0.3), linewidth=2)
    ax.plot(timespan, v, color=C_RED, linewidth=4)
    ax.plot(timespan, phi, '--', color=(0.4, 0.4, 0.4), linewidth=2)

    ax.set(xlabel='Time', ylabel='$v_i^w$', xlim=(min(timespan), max(timespan)), ylim=(0, 1))
           # xticks=(200, 400, 600, 800))#,  ylim=(-0.6, 0.6))

    plt.tight_layout()
    plt.show()


def plot_velocity_timeseries_multiple(nr, deltav, density, realization, agents, timespan):
    filename = f'ObservingAndInferring_29April2019_' \
               f'N42_NumberRatio_{nr}_packdens_{density}_delV_{deltav}_Fluc_0_Realization_{realization}.mat'
    filename = os.path.join(rootdir, filename)
    v_colors = {False: C_RED, True: C_BLUE}
    phi_colors = {False: C_RED_DARK, True: C_BLUE_DARK}

    mu = get_scale_factor_analytical(nr=nr, deltav=deltav, density=density)

    ad = AgentDynamics(filename)

    phi = mu * ad.compute_local_fields(voronoi=True, radial=True, relative=False, mean=True, cos=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    for agent in agents:
        v = moving_average(ad.vel, w=50, axis=2)[agent, 0, timespan]
        phi_ = moving_average(phi, w=50, axis=2)[agent, 0, timespan]
        # v = ad.vel[agent, 0, timespan]
        # phi = phi[agent, 0, timespan]

        ax.axhline(0, ls=':', color=(0.3, 0.3, 0.3), linewidth=2)
        ax.plot(timespan, v, color=v_colors[ad.labels[agent]], linewidth=4)
        ax.plot(timespan, phi_, '--', color=phi_colors[ad.labels[agent]], linewidth=2)

    ax.set(xlabel='Time', ylabel='$v_i^w$', xlim=(min(timespan), max(timespan)), ylim=(0, 1))
           # xticks=(200, 400, 600, 800))#,  ylim=(-0.6, 0.6))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    rootdir = '/Users/nabeel/Data/ObservingAndInferring/SimData'
    summaryfile = '/Users/nabeel/Data/output/update_2222_mean.npz'
    outdir = '/Users/nabeel/Data/output/update_2222_mean_kdes/'

    plot_velocity_timeseries_multiple(nr=4, deltav=1, density=0.45792, realization=98, agents=(39, 15), timespan=range(520, 580))
    # show_agent_snapshot(nr=4, deltav=1, density=0.45792, realization=98, focal_agents=[39, ], t=200, show_agent_ids=True)
    # plot_velocity_timeseries(nr=4, deltav=1, density=0.45792, realization=98, agent=39, timespan=range(400, 450))
    # plot_velocity_timeseries(nr=4, deltav=1, density=0.45792, realization=98, agent=10, timespan=range(400, 450))
    # plot_velocity_timeseries(nr=4, deltav=1, density=0.45792, realization=98, agent=15, timespan=range(400, 450))
    # plot_velocity_timeseries(nr=4, deltav=1, density=0.45792, realization=98, agent=17, timespan=range(400, 500))
    # plot_velocity_timeseries(nr=4, deltav=1, density=0.45792, realization=98, agent=23, timespan=range(400, 500))
    # plot_velocity_timeseries(nr=4, deltav=1, density=0.45792, realization=98, agent=34, timespan=range(400, 500))
    # plot_velocity_timeseries(nr=4, deltav=1, density=0.45792, realization=98, agent=35, timespan=range(400, 500))

    # os.makedirs(outdir, exist_ok=True)

    # nrs = [1, 2, 4, 7, 9, 12, 14, 16, 19, 21]
    # deltavs = [3, 2, 1, 0.75, 0.5, 0.1]
    # densities = [0.57706, 0.45792, 0.30847, 0.22182, ]
    # plot_nr_vs_misclassifications(density=0.45792, deltav=1, both=True)
    # plot_nr_vs_total_misclassifications(density=0.45792, deltav=1, both=True, mode='balanced')
    # plot_nr_vs_misclassifications(density=0.30847, deltav=2, both=False)
    # plot_nr_vs_misclassifications(density=0.45792, deltav=1, both=False)
    # plot_nr_vs_misclassifications(density=0.57706, deltav=0.75, both=False)
    #
    # plot_actual_vs_estimated_nr(density=0.30847, deltav=2, both=False)
    # plot_actual_vs_estimated_nr(density=0.45792, deltav=1, both=False)
    # plot_actual_vs_estimated_nr(density=0.57706, deltav=0.75, both=False)
    # plot_deltav_vs_misclassifications(nr=14, density=0.45792)

    # pm_plots_multipanel(densities=[0.45792, 0.30847, 0.57706],
    #                     deltavs=[1, 2, 0.75],
    #                     both=True)
    #
    # nr_plots_multipanel(densities=[0.45792, 0.30847, 0.57706],
    #                     deltavs=[1, 2, 0.75],
    #                     both=True)
    # batch_plot_kdes(
    #     nrs=[21],
    #     deltavs=[1, 2, 0.75],
    #     densities=[0.45792, 0.30847, 0.57706]
    # )
    # dir_pm = os.path.join(outdir, 'pm')
    # dir_pm_overall = os.path.join(outdir, 'pm_overall')
    # dir_est_nr = os.path.join(outdir, 'nr_estimated')
    # os.makedirs(dir_pm)
    # os.makedirs(dir_pm_overall)
    # os.makedirs(dir_est_nr)
    #
    # deltavs = [3, 2, 1, 0.75, 0.5, 0.1]
    # densities = [0.57706, 0.45792, 0.30847, 0.22182, ]
    #
    # for deltav in tqdm(deltavs):
    #     for density in tqdm(densities):
    #         file_pm = f'pm_rho_{density}_s0_{deltav}.png'
    #         file_pm_overall = f'pm_overall_rho_{density}_s0_{deltav}.png'
    #         file_est_nr = f'nr_est_rho_{density}_s0_{deltav}.png'
    #
    #         title = f'$\\rho = {density}, s_0 = {deltav}$'
    #
    #         plot_nr_vs_misclassifications(density=density, deltav=deltav, both=True,
    #                                       title=title,
    #                                       outfile=os.path.join(dir_pm, file_pm))
    #
    #         plot_nr_vs_total_misclassifications(density=density, deltav=deltav, both=True,
    #                                             title=title,
    #                                             outfile=os.path.join(dir_pm_overall, file_pm_overall))
    #
    #         plot_actual_vs_estimated_nr(density=density, deltav=deltav, both=True,
    #                                     title=title,
    #                                     outfile=os.path.join(dir_est_nr, file_est_nr))



