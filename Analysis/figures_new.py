import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.io import loadmat

from agent_dynamics import AgentDynamicsBase as AgentDynamics
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

rootdir = '/Users/nabeel/Data/ObservingAndInferring/SimData'
cachedir = '/Volumes/Backyard/Data/cache/expkernel_relvel'
# summaryfile = '/Users/nabeel/Data/output/classification_metrics_expkernel.npz'
summaryfile = '/Users/nabeel/Data/output/metrics_expkernel_relvel_nr.npz'
# summaryfile = '/Users/nabeel/Data/output/metrics_expkernel_voronoi_nr.npz'

def plot_nr_vs_misclassifications(density, deltav, both=True):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    data = np.load(summaryfile)
    nrs, deltavs, densities = data['nrs'], data['deltavs'], data['densities']
    nrs = np.array([nr / N for nr in nrs])
    jdeltav = np.argwhere(deltavs == deltav)
    kdensity = np.argwhere(densities == density)

    baseline_cms = data['baseline_cms'] / (1001 * 100)
    field_cms = data['field_cms_svm'] / (1001 * 100)

    # print(N * nrs)
    print(baseline_cms[:, jdeltav, kdensity, BLUE, RED].squeeze())
    print(N * (1 - nrs))

    print(baseline_cms[:, jdeltav, kdensity, RED, BLUE].squeeze())
    print(N * nrs)
    print(baseline_cms[:, jdeltav, kdensity, RED, BLUE].squeeze() / (N * nrs))

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

if __name__ == '__main__':
    plot_nr_vs_misclassifications(density=0.57706, deltav=0.75, both=True)
