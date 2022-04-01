import os

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from classify import DataClassifier, get_scale_factor_analytical
from agent_dynamics import f_poly, f_exp


def cache_all_data_parallel(rootdir, cachedir, nrs, densities, deltavs, f, realizations=100):
    Parallel(n_jobs=4)(delayed(DataClassifier)
                       (rootdir=rootdir, nr=nr, density=density, deltav=deltav, f=f,
                        realizations=realizations, tau=50, cachedir=cachedir)
                       for nr in nrs
                       for density in densities
                       for deltav in deltavs
                       )


def cache_all_data(rootdir, cachedir, nrs, densities, deltavs, f, realizations=100):
    """ Preload and cache all data for given sets of parameters. """
    for nr in tqdm(nrs, desc='nr'):
        for density in tqdm(densities, desc='density'):
            for deltav in tqdm(deltavs, desc='deltav'):
                DataClassifier(rootdir=rootdir, nr=nr, density=density, deltav=deltav, f=f,
                               realizations=realizations, cachedir=cachedir)


def compute_classification_metrics(rootdir, cachedir, outfile, nrs, deltavs, densities, f=f_exp):
    n_nrs, n_deltavs, n_densities = len(nrs), len(deltavs), len(densities)
    baseline_cms = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    mixvel_cms = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    field_cms_analytical = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    field_cms_svm_unbalanced = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    field_cms_svm_balanced = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    mus_unbalanced = np.zeros((n_nrs, n_deltavs, n_densities))
    mus_balanced = np.zeros((n_nrs, n_deltavs, n_densities))

    for inr, nr in enumerate(nrs):
        for jdeltav, deltav in enumerate(deltavs):
            for kdensity, density in enumerate(densities):
                dc = DataClassifier(rootdir=rootdir, cachedir=cachedir, nr=nr, density=density, deltav=deltav, f=f)
                mu_analytical = get_scale_factor_analytical(nr=nr, deltav=deltav, density=density, f=f)
                mu_unbalanced = dc.get_scale_factor_svm(balanced=False)
                mu_balanced = dc.get_scale_factor_svm(balanced=True)
                baseline_cms[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='baseline')
                mixvel_cms[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='mixvel')
                field_cms_analytical[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='field',
                                                                                             scale_factor=mu_analytical)
                field_cms_svm_unbalanced[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='field',
                                                                                                 scale_factor=mu_unbalanced)
                field_cms_svm_balanced[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='field',
                                                                                               scale_factor=mu_balanced)
                mus_unbalanced[inr, jdeltav, kdensity] = mu_unbalanced
                mus_balanced[inr, jdeltav, kdensity] = mu_balanced

    np.savez(outfile,
             baseline_cms=baseline_cms,
             mixvel_cms=mixvel_cms,
             field_cms_svm_balanced=field_cms_svm_balanced,
             field_cms_svm_unbalanced=field_cms_svm_unbalanced,
             field_cms_analytical=field_cms_analytical,
             mus_unbalanced=mus_unbalanced,
             mus_balanced=mus_balanced,
             nrs=nrs, deltavs=deltavs, densities=densities)


if __name__ == '__main__':
    rootdir = '/Users/nabeel/Data/ObservingAndInferring/SimData'
    # cachedir = '/Users/nabeel/Data/cache/update_2222_mean'
    cachedir = '/Volumes/Backyard/Data/cache/update_2222_mean'
    outdir = '/Users/nabeel/Data/output/'
    outfile = 'update_2222_mean.npz'

    os.makedirs(cachedir, exist_ok=True)

    # nrs = [1, 2, 4, 7, 9, 12, 14, 16, 19, 21]
    # deltavs = [3, 2, 1.5, 1, 0.75, 0.5, 0.25, 0.1, ]
    # densities = [0.57706, 0.45792, 0.3722, 0.30847, 0.25981, 0.22182, ]

    nrs = [1, 2, 4, 7, 9, 12, 14, 16, 19, 21]
    deltavs = [3, 2, 1, 0.75, 0.5, 0.1]
    densities = [0.57706, 0.45792, 0.30847, 0.22182, ]

    # deltavs = [0.75, ]
    # densities = [0.57706, ]
    f = lambda r: 1
    # cache_all_data_parallel(rootdir=rootdir, cachedir=cachedir, nrs=nrs, densities=densities, deltavs=deltavs, f=f)
    compute_classification_metrics(rootdir, cachedir,
                                   outfile=os.path.join(outdir, outfile),
                                   nrs=nrs, deltavs=deltavs, densities=densities)
