import os

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from classify import DataClassifier, get_scale_factor_analytical
from agent_dynamics import f_poly, f_exp


def cache_all_data_parallel(rootdir, cachedir, nrs, densities, deltavs, f, realizations=100):
    Parallel(n_jobs=4)(delayed(DataClassifier)
        (rootdir=rootdir, nr=nr, density=density, deltav=deltav, f=f, realizations=realizations, tau=50, cachedir=cachedir, scaling_mode='local')
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
    field_cms_analytical = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    field_cms_svm = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    mus = np.zeros((n_nrs, n_deltavs, n_densities))
    
    for inr, nr in enumerate(nrs):
        for jdeltav, deltav in enumerate(deltavs):
            for kdensity, density in enumerate(densities):
                dc = DataClassifier(rootdir=rootdir, cachedir=cachedir, nr=nr, density=density, deltav=deltav, f=f)
                # print(f'scale_factor_1: {dc.scale_factor_analytical}')
                # dc.scale_factor_analytical = get_scale_factor_analytical(nr=dc.nr, deltav=dc.deltav, density=dc.density, f=f)
                # print(f'scale_factor_2: {dc.scale_factor_analytical}')
                scale_factor = get_scale_factor_analytical(nr=nr, deltav=deltav, density=density, f=f)
                baseline_cms[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='baseline')
                field_cms_analytical[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='field', scale_factor=scale_factor)
                field_cms_svm[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='field', scale_factor=dc.scale_factor_svm)
                mus[inr, jdeltav, kdensity] = dc.scale_factor_svm

    np.savez(outfile, baseline_cms=baseline_cms, 
        field_cms_svm=field_cms_svm, field_cms_analytical=field_cms_analytical, mus=mus,
        nrs=nrs, deltavs=deltavs, densities=densities)


def compute_classification_metrics_rel(rootdir, cachedir, outfile, nrs, deltavs, densities, f=f_exp):
    n_nrs, n_deltavs, n_densities = len(nrs), len(deltavs), len(densities)
    baseline_cms = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    field_cms_analytical = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    field_cms_svm = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    mus = np.zeros((n_nrs, n_deltavs, n_densities))

    for inr, nr in enumerate(nrs):
        for jdeltav, deltav in enumerate(deltavs):
            for kdensity, density in enumerate(densities):
                dc = DataClassifier(rootdir=rootdir, cachedir=cachedir, nr=nr, density=density, deltav=deltav, f=f)
                # print(f'scale_factor_1: {dc.scale_factor_analytical}')
                # dc.scale_factor_analytical = get_scale_factor_analytical(nr=dc.nr, deltav=dc.deltav, density=dc.density, f=f)
                # print(f'scale_factor_2: {dc.scale_factor_analytical}')
                scale_factor = get_scale_factor_analytical(nr=nr, deltav=deltav, density=density, f=f)
                baseline_cms[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='baseline')
                field_cms_analytical[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='field_only',
                                                                                             scale_factor=scale_factor)
                field_cms_svm[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='field',
                                                                                      scale_factor=dc.scale_factor_svm)
                mus[inr, jdeltav, kdensity] = dc.scale_factor_svm

    np.savez(outfile, baseline_cms=baseline_cms,
             field_cms_svm=field_cms_svm, field_cms_analytical=field_cms_analytical, mus=mus,
             nrs=nrs, deltavs=deltavs, densities=densities)


if __name__ == '__main__':

    rootdir = '/Users/nabeel/Data/ObservingAndInferring/SimData'
    cachedir = '/Volumes/Backyard/Data/cache/expkernel_relative_voronoi'
    outdir = '/Users/nabeel/Data/output/'

    os.makedirs(cachedir, exist_ok=True)

    # nrs = [1, 2, 4, 7, 9, 12, 14, 16, 19, 21]
    # deltavs = [3, 2, 1.5, 1, 0.75, 0.5, 0.25, 0.1, ]
    # densities = [0.57706, 0.45792, 0.3722, 0.30847, 0.25981, 0.22182, ]

    # nrs = [9, 12, 19, 21]
    # deltavs = [3, 2.5, 2, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0.1, ]
    # densities = [0.45792, ]
    # # cache_all_data_parallel(rootdir=rootdir, cachedir=cachedir, nrs=nrs, densities=densities, deltavs=deltavs, f=f_exp)
    # compute_classification_metrics(rootdir, cachedir,
    #                                outfile=os.path.join(outdir, 'metrics_expkernel_relvel_deltav.npz'),
    #                                nrs=nrs, deltavs=deltavs, densities=densities)

    nrs = [1, 2, 4, 7, 9, 12, 14, 16, 19, 21]
    deltavs = [0.75, ]
    densities = [0.57706, ]
    # cache_all_data_parallel(rootdir=rootdir, cachedir=cachedir, nrs=nrs, densities=densities, deltavs=deltavs, f=f_exp)
    compute_classification_metrics_rel(rootdir, cachedir,
                                   outfile=os.path.join(outdir, 'metrics_expkernel_voronoi_nr.npz'),
                                   nrs=nrs, deltavs=deltavs, densities=densities)
