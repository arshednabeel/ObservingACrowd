import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from sklearn.svm import LinearSVC
from tqdm import tqdm, trange

from agent_dynamics import moving_average
from agent_dynamics import ADLaplacian as AgentDynamics


RED, BLUE = 0, 1


class DataClassifier(object):
    """ Data classifier object that loads data and gets classification results for one set of parameter values. 
    TODO: Add plotting functions if required
    """

    def __init__(self, rootdir, nr, density, deltav, mode, realizations=100, tau=50,
                 cachedir=None) -> None:
        
        self.rootdir = rootdir
        self.nr = nr
        self.density = density
        self.deltav = deltav
        self.realizations = realizations
        self.tau = tau
        self.mode = mode  # Mode for computing Laplacian

        self.initialized = False
        self.vel = None       # (n, 2, T, R)
        self.vel_avg = None   # (n, 2, T, R)
        self.psi = None       # (n, 2, T, R)
        self.psi_avg = None   # (n, 2, T, R)
        self.labels = None    # (n, R)

        self.load_data(cachedir)
        self.sanitize_data()

    def load_data(self, cachedir):

        if cachedir:
            cachefile = f'Cached_N42_NumberRatio_{self.nr}_packdens_{self.density}_delV_{self.deltav}_realizations_{self.realizations}.pkl'
            cachefile = os.path.join(cachedir, cachefile)
            if os.path.exists(cachefile):

                # print(f'Loading from cache.: {cachefile}')
                with open(cachefile, 'rb') as cache:
                    data = pickle.load(cache)
                self.vel = data['vel']
                self.vel_avg = data['vel_avg']
                self.psi = data['phi']
                self.psi_avg = data['phi_avg']
                self.labels = data['labels']
                return
            else:
                print(f'Processing: {cachefile}')

        for i in range(1, 1 + self.realizations): #, desc='Loading'):
            filename = f'ObservingAndInferring_29April2019_' \
                f'N42_NumberRatio_{self.nr}_packdens_{self.density}_delV_{self.deltav}_Fluc_0_Realization_{i}.mat'
            filename = os.path.join(self.rootdir, filename)
            ad = AgentDynamics(filename)
            
            if self.vel is None:
                self.vel = np.zeros((ad.n, 2, ad.T, self.realizations))
                self.vel_avg = np.zeros_like(self.vel)
                self.psi = np.zeros_like(self.vel)
                self.psi_avg = np.zeros_like(self.vel)
                self.labels = np.empty((ad.n, self.realizations), dtype=bool)

            psi = ad.get_velocity_field_laplacian(self.mode)

            # Sanity checks
            if (np.abs(ad.vel) > 1e5).any():
                print(f'WARNING: Sanity check (vel) failed for {filename}.')
            self.vel[:, :, :, i - 1] = ad.vel
            self.psi[:, :, :, i - 1] = psi
            self.vel_avg[:, :, :, i - 1] = moving_average(ad.vel, self.tau, axis=2)
            self.psi_avg[:, :, :, i - 1] = moving_average(psi, self.tau, axis=2)
            self.labels[:, i - 1] = ad.labels

        if cachedir:
            cachefile = f'Cached_N42_NumberRatio_{self.nr}_packdens_{self.density}_delV_{self.deltav}_realizations_{self.realizations}.pkl'
            print(f'Saving to cache: {cachefile}')
            cachefile = os.path.join(cachedir, cachefile)
            data = {
                'vel': self.vel, 'vel_avg': self.vel_avg, 
                'phi': self.psi, 'phi_avg': self.psi_avg,
                'labels': self.labels,
            }
            with open(cachefile, 'wb') as cache:
                pickle.dump(data, cache)

    def sanitize_data(self):
        """ Replace realizations where sanity check failed with previous realization. """

        for i in range(self.realizations):
            if (np.abs(self.vel[..., i]) > 1e3).any():
                print(f'Sanity check failed for: Nr={self.nr}, density={self.density}, deltav={self.deltav}, realization={i + 1}')
                self.vel[..., i] = self.vel[..., i - 1]
                self.psi[..., i] = self.psi[..., i - 1]
                self.vel_avg[..., i] = self.vel_avg[..., i - 1]
                self.psi_avg[..., i] = self.psi_avg[..., i - 1]

    def baseline_classifier(self):
        predictions = (self.vel_avg[:, 0, :, :] > 0).squeeze()
        return predictions

    def laplacian_classifier(self):
        predictions = (self.psi_avg[:, 0, :, :] > 0).squeeze()
        return predictions

    def get_confusion_matrix(self, method, scale_factor=None, timeseries=False, scaled=False):
        """ Get the confusion matrix using the given DataClassifier object 
        Params:
            dc: DataClassifier object with preloaded data
            method: 'baseline' or 'laplacian', denoting which classification method to use.
            scale_factor: Scale factor to use for field-based classification
            timeseries: If timeseries is true, keep the time axis and return a time-series
            of confusion matrices.
        Returns:
            Confusion matrix cm of predictions.
        """

        assert method in ['baseline', 'laplacian']

        labels = np.expand_dims(self.labels, 1)

        if method == 'baseline':
            predictions = self.baseline_classifier()
        else:  # method == 'laplacian'
            predictions = self.laplacian_classifier()

        if timeseries:
            axis = (0, 2)
            cm = np.zeros((2, 2, predictions.shape[1]), dtype=float)
            cm[RED, RED, :] = ((labels == RED) & (predictions == RED)).sum(axis=axis)
            cm[RED, BLUE, :] = ((labels == RED) & (predictions == BLUE)).sum(axis=axis)
            cm[BLUE, RED, :] = ((labels == BLUE) & (predictions == RED)).sum(axis=axis)
            cm[BLUE, BLUE, :] = ((labels == BLUE) & (predictions == BLUE)).sum(axis=axis)
        else:
            cm = np.zeros((2, 2), dtype=float)
            cm[RED, RED] = ((labels == RED) & (predictions == RED)).sum()
            cm[RED, BLUE] = ((labels == RED) & (predictions == BLUE)).sum()
            cm[BLUE, RED] = ((labels == BLUE) & (predictions == RED)).sum()
            cm[BLUE, BLUE] = ((labels == BLUE) & (predictions == BLUE)).sum()

        if scaled:
            cm[RED, :] = cm[RED, :] / cm[RED, :].sum()
            cm[BLUE, :] = cm[BLUE, :] / cm[BLUE, :].sum()

        return cm

    def show_distributions(self):
        fig, ax = plt.subplots(2, 1, figsize=(4, 8), sharex=True)

        n, _, T, r = self.vel.shape
        labels = np.expand_dims(self.labels, 1)
        labels = np.broadcast_to(labels, (n, T, r))
        mix_vel = self.vel[:, 0, ...].mean(axis=0)

        ax[0].axvline(0, color=(0.6, 0.6, 0.6), linewidth=1)
        ax[0].axvline(mix_vel[0, 0], color=(0.6, 0.6, 0.6), linewidth=1)
        ax[0].hist(self.vel_avg[:, 0, :, :][labels].flatten(), color='b', alpha=0.5, bins=100,
                   density=True, histtype='stepfilled')
        ax[0].hist(self.vel_avg[:, 0, :, :][~labels].flatten(), color='r', alpha=0.5, bins=100,
                   density=True, histtype='stepfilled')
        # ax[0].hist(mix_vel.flatten(), color='k', alpha=0.5, bins=100, density=True, histtype='stepfilled')
        ax[0].set(title='$v$ distribution', xlabel='$v_i$')#, xlim=(-1.5 * deltav, +1.5 * deltav))

        ax[1].axvline(0, color=(0.6, 0.6, 0.6), linewidth=1)
        ax[1].hist(self.psi_avg[:, 0, :, :][labels].flatten(), color='b', alpha=0.5, bins=100,
                   density=True, histtype='stepfilled')
        ax[1].hist(self.psi_avg[:, 0, :, :][~labels].flatten(), color='r', alpha=0.5, bins=100,
                   density=True, histtype='stepfilled')
        ax[1].set(title='$\\psi$ distribution', xlabel='$\\phi_i$')#, xlim=(-1.5 * deltav, +1.5 * deltav))

        fig.suptitle(f'$N_r = {nr}, s_0 = {deltav}, \\rho = {density}$')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    rootdir = '/Users/nabeel/Data/ObservingAndInferring/SimData'
    nr = 4
    # density = 0.57706
    density = 0.45792
    # density = 0.22182
    deltav = 0.75
    # deltav = 2
    dc = DataClassifier(rootdir=rootdir, nr=nr, density=density, deltav=deltav, mode='angle',
                        realizations=10, cachedir=None)
    dc.show_distributions()

    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.scatter(dc.vel[:, 0, 0, :].squeeze(), dc.vel[:, 0, :, :].squeeze())

    #
    # # fig, ax = plt.subplots(figsize=(7, 5))
    # # mix_vel = dc.vel[:, 0, ...].mean(axis=0)
    # # v = dc.vel_avg[:, 0, :, 0].T
    # # ax.plot(v, alpha=0.5, color='r')
    # # ax.plot(v.mean(axis=1), color='k')
    # # ax.set(ylim=[0.6, 0.8])
    # #
    # # # ax.plot(mix_vel, alpha=0.5, color='k')
    # # plt.show()
    #
    # # p0 = dc.baseline_classifier()
    # # p1 = dc.local_field_classifier()
    #
    # # labels = np.expand_dims(dc.labels, 1)
    #
    # # print(f'Baseline: {(p0 == labels).sum()} correct')
    # # print(f'Field-based: {(p1 == labels).sum()} correct')
    # # print(f'Mismatch: {(p0 != p1).sum()}')
    #
    # # dc.scale_factor = 0.2
    # dc.scatterplot(scale_factor=1/6, scale_factor_2=dc.scale_factor_svm)
    # dc.ideal_scatterplot()

    # print(dc.mixture_velocity_classifier())
    # dc.show_distributions(scale_factor=1/6)#dc.scale_factor_analytical)
    # print(dc.get_confusion_matrix(method='field_only', scaled=True))



