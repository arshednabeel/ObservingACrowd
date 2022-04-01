import os

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from classify_laplacian import DataClassifier


def cache_all_data_parallel(rootdir, cachedir, nrs, densities, deltavs, mode, realizations=100):
    Parallel(n_jobs=4)(delayed(DataClassifier)
                       (rootdir=rootdir, nr=nr, density=density, deltav=deltav, mode=mode,
                        realizations=realizations, tau=50, cachedir=cachedir)
                       for nr in nrs
                       for density in densities
                       for deltav in deltavs
                       )


def cache_all_data(rootdir, cachedir, nrs, densities, deltavs, mode, realizations=100):
    """ Preload and cache all data for given sets of parameters. """
    for nr in tqdm(nrs, desc='nr'):
        for density in tqdm(densities, desc='density'):
            for deltav in tqdm(deltavs, desc='deltav'):
                DataClassifier(rootdir=rootdir, nr=nr, density=density, deltav=deltav, mode=mode,
                               realizations=realizations, cachedir=cachedir)


def compute_classification_metrics(rootdir, cachedir, outfile, nrs, deltavs, densities, mode):
    n_nrs, n_deltavs, n_densities = len(nrs), len(deltavs), len(densities)
    baseline_cms = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))
    laplacian_cms = np.zeros((n_nrs, n_deltavs, n_densities, 2, 2))

    for inr, nr in enumerate(nrs):
        for jdeltav, deltav in enumerate(deltavs):
            for kdensity, density in enumerate(densities):
                dc = DataClassifier(rootdir=rootdir, cachedir=cachedir, nr=nr, density=density, deltav=deltav, mode=mode)
                # print(f'scale_factor_1: {dc.scale_factor_analytical}')
                # dc.scale_factor_analytical = get_scale_factor_analytical(nr=dc.nr, deltav=dc.deltav, density=dc.density, f=f)
                # print(f'scale_factor_2: {dc.scale_factor_analytical}')

                baseline_cms[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='baseline')
                laplacian_cms[inr, jdeltav, kdensity, :, :] = dc.get_confusion_matrix(method='laplacian')

    np.savez(outfile, baseline_cms=baseline_cms, laplacian_cms=laplacian_cms,
             nrs=nrs, deltavs=deltavs, densities=densities)


if __name__ == '__main__':
    rootdir = '/Users/nabeel/Data/ObservingAndInferring/SimData'
    cachedir = '/Users/nabeel/Data/cache/laplacian_angle'
    # cachedir = '/Volumes/Backyard/Data/cache/update_2222'
    outdir = '/Users/nabeel/Data/output/'
    outfile = 'laplacian_angle.npz'

    os.makedirs(cachedir, exist_ok=True)

    # nrs = [1, 2, 4, 7, 9, 12, 14, 16, 19, 21]
    # deltavs = [3, 2, 1.5, 1, 0.75, 0.5, 0.25, 0.1, ]
    # densities = [0.57706, 0.45792, 0.3722, 0.30847, 0.25981, 0.22182, ]

    nrs = [1, 2, 4, 7, 9, 12, 14, 16, 19, 21]
    # deltavs = [3, 2, 1, 0.75, 0.5, 0.1]
    # densities = [0.57706, 0.45792, 0.30847, 0.22182, ]

    deltavs = [0.75, ]
    densities = [0.57706, ]
    cache_all_data_parallel(rootdir=rootdir, cachedir=cachedir,
                            nrs=nrs, densities=densities, deltavs=deltavs, mode='angle')
    compute_classification_metrics(rootdir, cachedir,
                                   outfile=os.path.join(outdir, outfile),
                                   nrs=nrs, deltavs=deltavs, densities=densities, mode='angle')
