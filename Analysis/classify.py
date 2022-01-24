import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from sklearn.svm import LinearSVC
from tqdm import tqdm, trange

from agent_dynamics import AgentDynamics, moving_average, f_exp, f_poly

RED, BLUE = 0, 1

def get_scale_factor_analytical(nr, deltav, density, f=f_exp):
    """ Function that computes scaling factor analytically. """

    def Pk(k, nr):
        p = nr / 42
        return binom.pmf(k, 6, p)

    sigma_s = np.sum([Pk(k, nr) * np.abs(6 - 2 * k) for k in range(0, 7)])
    L_s = 1 / (np.sqrt(density))
    f_s = f(L_s)
    mu = 1 / (f_s * sigma_s)

    return mu

def get_sigma_s(nr):
    def Pk(k, nr):
        p = nr / 42
        return binom.pmf(k, 6, p)

    sigma_s = np.sum([Pk(k, nr) * np.abs(6 - 2 * k) for k in range(0, 7)])
    return sigma_s

class DataClassifier(object):
    """ Data classifier object that loads data and gets classification results for one set of parameter values. 
    TODO: Add plotting functions if required
    """

    def __init__(self, rootdir, nr, density, deltav, f=f_exp, realizations=100, tau=50, cachedir=None, scaling_mode='global') -> None:
        assert scaling_mode in ['global', 'local'], "Scaling mode should be 'global' or 'local'."
        
        self.rootdir = rootdir
        self.nr = nr
        self.density = density
        self.deltav = deltav
        self.realizations = realizations
        self.tau = tau
        self.f = f

        self.initialized = False
        self.vel = None
        self.vel_avg = None
        self.phi = None
        self.phi_avg = None
        self.labels = None
        self.scaling_mode = scaling_mode

        self.load_data(cachedir)
        self.sanitize_data()
        # self.scale_factor_minmax = self.get_scale_factor()
        self.scale_factor_svm = self.get_scale_factor_svm()
        self.scale_factor_analytical = get_scale_factor_analytical(
            nr=self.nr, deltav=self.deltav, density=self.density) if scaling_mode == 'global' else 1

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
                self.phi = data['phi']
                self.phi_avg = data['phi_avg']
                self.labels = data['labels']
                self.scaling_mode = data.get('scaling_mode', 'global')
                return
            else:
                print(f'Processing: {cachefile}')

        sigma_s_global = get_sigma_s(nr=self.nr)
        for i in range(1, 1 + self.realizations): #, desc='Loading'):
            filename = f'ObservingAndInferring_29April2019_' \
                f'N42_NumberRatio_{self.nr}_packdens_{self.density}_delV_{self.deltav}_Fluc_0_Realization_{i}.mat'
            filename = os.path.join(self.rootdir, filename)
            ad = AgentDynamics(filename)
            
            if self.vel is None:
                self.vel = np.zeros((ad.n, 2, ad.T, self.realizations))
                self.vel_avg = np.zeros_like(self.vel)
                self.phi = np.zeros_like(self.vel)
                self.phi_avg = np.zeros_like(self.vel)
                self.labels = np.empty((ad.n, self.realizations), dtype=bool)

            phi = ad.compute_radial_local_fields(self.f, scaling_mode=self.scaling_mode, sigma_s_global=sigma_s_global)

            # Sanity checks
            if (np.abs(ad.vel) > 1e5).any():
                print(f'WARNING: Sanity check (vel) failed for {filename}.')
            # elif (np.abs(phi) > 1e5).any():
            #     print(f'WARNING: Sanity check (phi) failed for {filename}.')

            self.vel[:, :, :, i - 1] = ad.vel
            self.phi[:, :, :, i - 1] = phi
            self.vel_avg[:, :, :, i - 1] = moving_average(ad.vel, self.tau, axis=2)
            self.phi_avg[:, :, :, i - 1] = moving_average(phi, self.tau, axis=2)
            self.labels[:, i - 1] = ad.labels

        if cachedir:
            cachefile = f'Cached_N42_NumberRatio_{self.nr}_packdens_{self.density}_delV_{self.deltav}_realizations_{self.realizations}.pkl'
            print(f'Saving to cache: {cachefile}')
            cachefile = os.path.join(cachedir, cachefile)
            data = {
                'vel': self.vel, 'vel_avg': self.vel_avg, 
                'phi': self.phi, 'phi_avg': self.phi_avg, 
                'labels': self.labels,
                'scaling_mode': self.scaling_mode
            }
            with open(cachefile, 'wb') as cache:
                pickle.dump(data, cache)

    def sanitize_data(self):
        """ Replace realizations where sanity check failed with previous realization. """

        for i in range(self.realizations):
            if (np.abs(self.vel[..., i]) > 1e3).any():
                print(f'Sanity check failed for: Nr={self.nr}, density={self.density}, deltav={self.deltav}, realization={i + 1}')
                self.vel[..., i] = self.vel[..., i - 1]
                self.phi[..., i] = self.phi[..., i - 1]
                self.vel_avg[..., i] = self.vel_avg[..., i - 1]
                self.phi_avg[..., i] = self.phi_avg[..., i - 1]


    def get_scale_factor(self):      
        q = 1e-6
        phi_min = np.quantile(self.phi_avg[:, 0, ...], q, interpolation='higher')
        phi_max = np.quantile(self.phi_avg[:, 0, ...], 1 - q, interpolation='lower')
        vel_min = np.quantile(self.vel_avg[:, 0, ...], q, interpolation='higher')
        vel_max = np.quantile(self.vel_avg[:, 0, ...], 1 - q, interpolation='lower')
        
        scale_factor = np.min((np.abs(vel_max / phi_max), np.abs(vel_min / phi_min)))

        return scale_factor

    def get_scale_factor_svm(self):
        n, _, T, r = self.vel.shape
        clf = LinearSVC(dual=False, fit_intercept=False) #, class_weight='balanced')

        vel = self.vel_avg[:, 0, ...].flatten()
        phi = self.phi_avg[:, 0, ...].flatten()
        labels = np.expand_dims(self.labels, 1)
        labels = np.broadcast_to(labels, (n, T, r))
        X = np.column_stack((vel, phi))
        X = X[np.isfinite(phi), :]
        y = labels.flatten()[np.isfinite(phi)]

        clf.fit(X, y)
        
        scale_factor = - clf.coef_[0, 1] / clf.coef_[0, 0]

        return scale_factor

    def baseline_classifier(self):
        predictions = (self.vel_avg[:, 0, :, :] > 0).squeeze()
        return predictions

    def local_field_classifier(self, scale_factor):
        """ 
        Returns:
            predictions: (n, T, r) boolean array of predicted labels
        """

        predictions = (self.vel_avg[:, 0, :, :] > scale_factor * self.phi_avg[:, 0, :, :]).squeeze()
        return predictions

    def local_field_svm_classifer(self):
        n, _, T, r = self.vel.shape
        clf = LinearSVC(dual=False, fit_intercept=False, class_weight='balanced')

        vel = self.vel_avg[:, 0, ...].flatten()
        phi = self.phi_avg[:, 0, ...].flatten()
        labels = np.expand_dims(self.labels, 1)
        labels = np.broadcast_to(labels, (n, T, r))
        X = np.column_stack((vel, phi))
        y = labels.flatten()

        # print(f'X.shape: {X.shape}')
        # print(f'y.shape: {y.shape}')
        
        clf.fit(X, y)

        print(f'mean accuracy of SVM fit: {clf.score(X, y)}')
        predictions = clf.predict(X)
        predictions = predictions.reshape(labels.shape)
        return clf, predictions, X, y, clf.decision_function(X)

    def get_confusion_matrix(self, method: str = 'field', scale_factor=None, timeseries=False):
        """ Get the confusion matrix using the given DataClassifier object 
        Params:
            dc: DataClassifier object with preloaded data
            method: 'baseline' or 'field', denoting which classification method to use.
            scale_factor: Scale factor to use for field-based classification
            timeseries: If timeseries is true, keep the time axis and return a time-series
            of confusion matrices.
        Returns:
            Confusion matrix cm of predictions.
        """

        assert method in ['baseline', 'field'], "method should be 'baseline' or 'field'"
        if scale_factor is None:
            scale_factor = self.scale_factor_svm

        labels = np.expand_dims(self.labels, 1)
        if method == 'baseline':
            predictions = self.baseline_classifier()
        else:
            predictions = self.local_field_classifier(scale_factor)

        if timeseries:
            axis = (0, 2)
            cm = np.zeros((2, 2, predictions.shape[1]), dtype=int)
            cm[RED, RED, :] = ((labels == RED) & (predictions == RED)).sum(axis=axis)
            cm[RED, BLUE, :] = ((labels == RED) & (predictions == BLUE)).sum(axis=axis)
            cm[BLUE, RED, :] = ((labels == BLUE) & (predictions == RED)).sum(axis=axis)
            cm[BLUE, BLUE, :] = ((labels == BLUE) & (predictions == BLUE)).sum(axis=axis)
        else:
            cm = np.zeros((2, 2), dtype=int)
            cm[RED, RED] = ((labels == RED) & (predictions == RED)).sum()
            cm[RED, BLUE] = ((labels == RED) & (predictions == BLUE)).sum()
            cm[BLUE, RED] = ((labels == BLUE) & (predictions == RED)).sum()
            cm[BLUE, BLUE] = ((labels == BLUE) & (predictions == BLUE)).sum()

        return cm

    def get_misclassifications(self, scale_factor):
        """ Get number of misclassifications with confidence intervals
        Returns:
            nm_mean: Mean number of misclassifications
            nm_confint: 95 % confidence interval for number of misclassifications
        """
        labels = np.expand_dims(self.labels, 1)
        predictions = self.local_field_classifier(scale_factor)
        n, T, r = predictions.shape
        nm_blue = ((labels == BLUE) & (predictions == RED)).sum(axis=0)   # shape (T, r)
        nm_blue_mean = nm_blue.mean()
        nm_blue_se = nm_blue.std() / np.sqrt(T * r)
        nm_blue_confint = (nm_blue_mean - 1.96 * nm_blue_se, nm_blue_mean + 1.96 * nm_blue_se)

        nm_red = ((labels == RED) & (predictions == BLUE)).sum(axis=0)  # shape (T, r)
        nm_red_mean = nm_red.mean()
        nm_red_se = nm_red.std() / np.sqrt(T * r)
        nm_red_confint = (nm_red_mean - 1.96 * nm_red_se, nm_red_mean + 1.96 * nm_red_se)

        return nm_blue_mean, nm_red_mean, nm_blue_confint, nm_red_confint

    def get_misclassification_array(self, scale_factor):
        """ Get number of misclassifications as an array.
        Returns:
            nm_blue: (T, r) array of misclassifications for BLUE group.
            nm_red: (T, r) array of misclassifications for RED group.
        """

        labels = np.expand_dims(self.labels, 1)
        predictions = self.local_field_classifier(scale_factor)
        nm_blue = ((labels == BLUE) & (predictions == RED)).sum(axis=0)  # shape (T, r)
        nm_red = ((labels == RED) & (predictions == BLUE)).sum(axis=0)  # shape (T, r)
        return nm_blue, nm_red

    def scatterplot(self, scale_factor, filename=None, color1='b', color2='r', show_title=False, scale_factor_2=None):
        n, _, T, r = self.vel.shape
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        labels = np.expand_dims(self.labels, 1)
        labels = np.broadcast_to(labels, (n, T, r))
        # print(f'labels.dtype: {labels.dtype}')
        # print(f'labels.shape: {labels.shape}')
        # print(f'self.labels.dtype: {self.labels.dtype}')
        if show_title:
            fig.suptitle(f'Nr = {self.nr}, deltav = {self.deltav}, density = {self.density}, scale factor = {scale_factor}')
        ax.scatter(self.vel_avg[:, 0, :, :][labels].flatten(), 
            self.phi_avg[:, 0, :, :][labels].flatten(), alpha=0.02, facecolor=color1)
        ax.scatter(self.vel_avg[:, 0, :, :][~labels].flatten(), 
            self.phi_avg[:, 0, :, :][~labels].flatten(), alpha=0.02, facecolor=color2)

        ax.axline(xy1=(0, 0), slope=(1 / scale_factor), linewidth=2, color='k')
        if scale_factor_2:
            ax.axline(xy1=(0, 0), slope=(1 / scale_factor_2), linewidth=2, color=(0.5, 0.5, 0.5))
        ax.set_xlabel('$v$')
        ax.set_ylabel('$\\varphi$')
        ax.set(xticks=[-self.deltav, 0, self.deltav], yticks=[-self.deltav, 0, self.deltav])
        ax.axis('equal')
        # ax.set(xlim=(-2 * self.deltav, +2 * self.deltav), ylim=(-2 * self.deltav, +2 * self.deltav))
        ax.grid(True)

        plt.tight_layout()
        if filename:
            fig.savefig(filename)
            plt.close(fig)
        else:
            plt.show()


if __name__ == '__main__':
    rootdir = '/Users/nabeel/Data/ObservingAndInferring/SimData'
    nr = 16
    density = 0.45792
    deltav = 1
    f = lambda r: np.exp(-((r/3) ** 2))
    # f = lambda r: 1

    # cachedir = '/Volumes/Backyard/Data/cache/expkernel'
    # if not os.path.exists(cachedir):
    #     os.makedirs(cachedir)

    dc = DataClassifier(rootdir=rootdir, nr=nr, density=density, deltav=deltav, f=f,
        realizations=10, cachedir=None)

    # p0 = dc.baseline_classifier()
    # p1 = dc.local_field_classifier()

    # labels = np.expand_dims(dc.labels, 1)

    # print(f'Baseline: {(p0 == labels).sum()} correct')
    # print(f'Field-based: {(p1 == labels).sum()} correct')
    # print(f'Mismatch: {(p0 != p1).sum()}')

    # dc.scale_factor = 0.2
    dc.scatterplot(scale_factor=dc.scale_factor_analytical)



