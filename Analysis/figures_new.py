import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.io import loadmat

from agent_dynamics import AgentDynamics
from classify import DataClassifier, f_exp, get_scale_factor_analytical

RED, BLUE = 0, 1
N = 42

plt.rcParams.update(
    {
        'font.size': 28,
    }
)

# Colors
C_RED = (178/255, 24/255, 43/255)
C_BLUE = (33/255, 102/255, 172/255)

C_RED_DARK = (103/255, 0/255, 31/255)
C_BLUE_DARK = (5/255, 48/255, 97/255)

# rootdir = '/Users/nabeel/Data/ObservingAndInferring/SimData'
# cachedir = '/Volumes/Backyard/Data/cache/expkernel_relvel'
# summaryfile = '/Users/nabeel/Data/output/update_7222_cosweight.npz'
summaryfile = '/Users/nabeel/Data/output/update_2222.npz'
# summaryfile = '/Users/nabeel/Data/output/constkernel_vor1_rel0_rad1_mean1_new.npz'

def plot_nr_vs_misclassifications(density, deltav, both=True):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    data = np.load(summaryfile)
    nrs, deltavs, densities = data['nrs'], data['deltavs'], data['densities']
    nrs = np.array([nr / N for nr in nrs])
    jdeltav = np.argwhere(deltavs == deltav)
    kdensity = np.argwhere(densities == density)

    baseline_cms = data['baseline_cms'] / (1001 * 100)
    # mixvel_cms = data['mixvel_cms'] / (1001 * 100)
    field_cms = data['field_cms_svm'] / (1001 * 100)

    ax.plot(nrs,
            baseline_cms[:, jdeltav, kdensity, RED, BLUE].squeeze() / (N * nrs),
            'o-', color=C_RED,
            label='C-I (Smaller Group)',
            linewidth=3, markersize=10)
    ax.plot(nrs,
            baseline_cms[:, jdeltav, kdensity, BLUE, RED].squeeze() / (N * (1 - nrs)),
            'o-', color=C_BLUE,
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
    ax.set(xlabel='$N_r$', ylabel='Prob. Misclassification')

    plt.tight_layout()
    plt.show()


def plot_deltav_vs_misclassifications(nr, density, both=True):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    data = np.load(summaryfile)
    nrs, deltavs, densities = data['nrs'], data['deltavs'], data['densities']

    inr = np.argwhere(nrs == nr)
    kdensity = np.argwhere(densities == density)

    baseline_cms = data['baseline_cms'] / (1001 * 100)
    field_cms = data['field_cms_analytical'] / (1001 * 100)

    nr = nr / N
    ax.plot(deltavs, baseline_cms[inr, :, kdensity, RED, BLUE].squeeze() / (N * nr),
            'o-', color=C_RED,
            label='C-I (Smaller Group)',
            linewidth=3, markersize=10)

    if both:
        ax.plot(deltavs, field_cms[inr, :, kdensity, RED, BLUE].squeeze() / (N * nr),
                's-', color=C_RED_DARK,
                label='C-II (Smaller Group)',
                linewidth=3, markersize=10)

    if nr != 21:
        ax.plot(deltavs, baseline_cms[inr, :, kdensity, BLUE, RED].squeeze() / (N * (1 - nr)),
                'o-', color=C_BLUE,
                label='C-I (Larger Group)',
                linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=2)

    if nr != 21 and both:
        ax.plot(deltavs, field_cms[inr, :, kdensity, BLUE, RED].squeeze() / (N * (1 - nr)),
                's-', color=C_BLUE_DARK,
                label='C-II (Larger Group)',
                linewidth=3, markersize=10, markerfacecolor='w', markeredgewidth=2)

    ax.set(xlabel='$s_0$', ylabel='Misclassifications')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # nrs = [1, 2, 4, 7, 9, 12, 14, 16, 19, 21]
    # deltavs = [3, 2, 1, 0.75, 0.5, 0.1]
    # densities = [0.57706, 0.45792, 0.30847, 0.22182, ]
    plot_nr_vs_misclassifications(density=0.57706, deltav=0.75, both=True)
    # plot_deltav_vs_misclassifications(nr=14, density=0.45792)
