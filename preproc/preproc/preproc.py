import mne
import numpy as np
import os
import scipy as sp
import sys

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Shrinkage
from pyriemann.tangentspace import TangentSpace

from transformer import PFD, HFD, Hurst
from pyriemann.utils.covariance import cospectrum

def _nextpow2(i):
    """Find next power of 2."""
    n = 1
    while n < i:
        n *= 2
    return n

def mad(data):
    """Median absolute deviation"""
    m = np.median(np.abs(data - np.median(data)))
    return m

class Windower(BaseEstimator, TransformerMixin):
    """Window."""

    def __init__(self, window=60, overlap=0, srate=200, unit='microvolts'):
        """Init."""
        self.window = window
        self.overlap = overlap
        self.srate = 200
        if unit == 'microvolts':
            self.multiplier = 1E6 
        else:
            self.multiplier = 1
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        wi = int(self.window * self.srate)
        ov = int(self.overlap * wi)
        out = []
        for x in X:
            nSamples = x.shape[1]
            ind = list(range(0, nSamples - wi + 1, wi - ov))
            for idx in ind:
                sl = slice(idx, idx + wi)
                out.append(x[:, sl]*self.multiplier)
        return np.array(out)


class MinMax(BaseEstimator, TransformerMixin):
    """Withening."""

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            tmp = [np.min(x), np.max(x)]
            out.append(tmp)
        return np.array(out)


from pyriemann.utils.base import invsqrtm
class Whitening(BaseEstimator, TransformerMixin):
    """Withening."""

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            if np.sum(x) != 0:
                cov = np.cov(x)
                W = invsqrtm(cov)
                tmp = np.dot(W.T, x)
            else:
                tmp = x
            out.append(tmp)
        return np.array(out)

from sklearn.decomposition import PCA
class ApplyPCA(BaseEstimator, TransformerMixin):
    """Withening."""

    def __init__(self, n_components=2):
        """Init."""
        self.n_components = n_components
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            tmp = PCA(self.n_components).fit_transform(x.T).T
            out.append(tmp)
        return np.array(out)

class Slicer(BaseEstimator, TransformerMixin):
    """Window."""

    def __init__(self, tmin=0, tmax=60, srate=200):
        """Init."""
        self.tmin = tmin
        self.tmax = tmax
        self.srate = srate
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        tmin = int(self.tmin * self.srate)
        tmax = int(self.tmax * self.srate)
        sl = slice(tmin, tmax)
        out = []
        for x in X:
            out.append(x[:, sl])
        return np.array(out)

class RemoveDropped(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            good_idx = (np.sum(x**2, 0) != 0)
            if np.sum(good_idx)==120000:  # change from 240000 to 120000 (10*60*200)
                # if data only contains dropped sample, pass it as it
                # to avoid passing empty array
                out.append(x)
            else:
                # else remove dropped packet
                out.append(x[:, good_idx])
        return np.array(out)


class IsEmpty(BaseEstimator, TransformerMixin):
    """Is the data empty ?"""

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            if np.sum(x) == 0:
                # if data only contains dropped sample, pass it as it
                # to avoid passing empty array
                out.append([1])
            else:
                # else remove dropped packet
                out.append([0])
        return np.array(out)

class InterpolateSpikes(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self,th=20):
        """Init."""
        self.th = th

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and rinerpolates dropped sample
        """
        out = []
        for x in X:
            avg_ref =  np.mean(x, 0)
            m = mad(avg_ref)
            no_spikes = avg_ref < (self.th * m)
            #print (np.sum(no_spikes), m)
            if m!=0:
                indices = np.arange(len(avg_ref))
                for ii, ch in enumerate(x):
                    x[ii] = np.interp(indices, indices[no_spikes], ch[no_spikes])
            out.append(x)
        return np.array(out)

class Useless(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self, nsamples=2):
        """Init."""
        self.nsamples = nsamples

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and rinerpolates dropped sample
        """
        out = []
        for x in X:
            tmp = x[:, 0:self.nsamples].flatten()
            out.append(tmp)
        return np.array(out)
#### coherence

from scipy.signal import filtfilt, butter

class FrequenctialFilter(BaseEstimator, TransformerMixin):
    """Withening."""

    def __init__(self, order=4, freqs=[4, 15], ftype='bandpass'):
        """Init."""
        self.order = order
        self.freqs = freqs
        self.ftype = ftype

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        b, a = butter(self.order, np.array(self.freqs) / 200., self.ftype)
        out = filtfilt(b, a, X, axis=-1)
        return out


from scipy.signal import hann, welch

from scipy.signal import argrelextrema

def find_peak(c, fr, order=5, max_peak=3):
    out = []
    for ci in c:
        tmp = []
        for ch in ci.T:
            a = argrelextrema(ch, np.greater, order=order)[0]
            if len(a) < max_peak:
                a = np.r_[a, [0] * (max_peak - len(a))]

            tmp.extend(list(fr[a[0:max_peak]]))
        out.append(np.array(tmp))
    return np.array(out)

def peak_freq(data, window=256, fs=200, overlap=0., ignore_dropped=False,
               frequencies=[6, 20]):

    nChan, nSamples = data.shape
    noverlap = int(overlap * window)
    windowVals = hann(window)

    # get the corresponding indices for custom frequencies
    freqs = np.fft.fftfreq(window, d=1./fs)[:int(window/2)]
    idx_freqs = []
    idx_freqs.append((freqs < frequencies[0]) | (freqs > frequencies[1]))

    ind = list(range(0, nSamples - window + 1, window-noverlap))

    numSlices = len(ind)
    slices = range(numSlices)

    Slices = []
    for iSlice in slices:
        thisSlice = data[:, ind[iSlice]:ind[iSlice] + window]
        if np.sum(np.sum(thisSlice**2, axis=0)>0):
            freqs, thisfft = welch(thisSlice, fs=200, nfft=int(window/2))
            Slices.append(thisfft.T)
    if len(Slices) > 0:
        Slices = np.array(Slices)
        a = find_peak(Slices, freqs, order=5, max_peak=3)
    else:
        a = np.nan
    return a


def slidingFFT(data, window=256, fs=200, overlap=0., ignore_dropped=False,
                frequencies=None, aggregate=True, phase=False):

    nChan, nSamples = data.shape
    noverlap = int(overlap * window)
    windowVals = hann(window)

    # get the corresponding indices for custom frequencies
    freqs = np.fft.fftfreq(window, d=1./fs)[:int(window/2)]
    idx_freqs = []
    if frequencies is not None:
        for fr in frequencies:
            tmp = (freqs >= fr[0]) & (freqs < fr[1])
            idx_freqs.append(np.where(tmp)[0])
            numFreqs = len(idx_freqs)
    else:
        numFreqs = len(freqs)
    # get the indices of dropped data
    if ignore_dropped:
        dropped = (np.sum(data**2, 0) == 0)

    ind = list(range(0, nSamples - window + 1, window-noverlap))

    numSlices = len(ind)
    slices = range(numSlices)
    Slices = np.zeros((numSlices, numFreqs, nChan), dtype=np.complex_)
    for iSlice in slices:
        sl = slice(ind[iSlice], ind[iSlice] + window)
        if ignore_dropped:
            if np.sum(dropped[sl]) > 0:
                continue

        thisSlice = data[:, sl]
        thisSlice = windowVals*thisSlice
        thisfft = np.fft.fft(thisSlice).T
        if frequencies is None:
            Slices[iSlice] = thisfft[1:(int(window/2) + 1)]
        else:
            for fr, idx in enumerate(idx_freqs):
                Slices[iSlice, fr, :] = thisfft[idx].mean(0)

    Slices = Slices.transpose(0, 2, 1)
    if aggregate:
        Slices = np.concatenate(Slices.transpose(1, 2, 0), axis=0)
    else:
        Slices = Slices.transpose(2, 1, 0)

    if phase:
        Slices = np.arctan2(np.imag(Slices), np.real(Slices))
    else:
        Slices = np.abs(Slices)

    return Slices


class SlidingFFT(BaseEstimator, TransformerMixin):

    """Slinding FFT
    """

    def __init__(self, window=256, overlap=0.5, fs=200,
                 frequencies=None, aggregate=True, ignore_dropped=False,
                 phase=False):
        """Init."""
        self.window = window
        self.overlap = overlap
        self.fs = fs
        self.frequencies = frequencies
        self.aggregate = aggregate
        self.ignore_dropped = ignore_dropped
        self.phase = phase

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : CospCovariances instance
            The CospCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the cospectral covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of covariance matrices for each trials and for each
            frequency bin.
        """
        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            S = slidingFFT(X[i], window=self.window, fs=self.fs,
                            overlap=self.overlap, frequencies=self.frequencies,
                            aggregate=self.aggregate, phase=self.phase,
                            ignore_dropped=self.ignore_dropped)
            out.append(S)

        return S


def coherences(data, window=256, fs=200, overlap=0., ignore_dropped=False,
               frequencies=[[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]], # , [90, 170]
               aggregate=False, transpose=False, normalize=True):

    nChan, nSamples = data.shape
    noverlap = int(overlap * window)
    windowVals = hann(window)

    # get the corresponding indices for custom frequencies
    freqs = np.fft.fftfreq(window, d=1./fs)[:int(window/2)]
    idx_freqs = []
    if frequencies is not None:
        for fr in frequencies:
            tmp = (freqs >= fr[0]) & (freqs < fr[1])
            idx_freqs.append(np.where(tmp)[0])
            numFreqs = len(idx_freqs)
    else:
        numFreqs = len(freqs)
    # get the indices of dropped data
    if ignore_dropped:
        dropped = (np.sum(data**2, 0) == 0)

    ind = list(range(0, nSamples - window + 1, window-noverlap))

    numSlices = len(ind)
    FFTSlices = {}
    FFTConjSlices = {}
    Pxx = {}
    slices = range(numSlices)
    normVal = np.linalg.norm(windowVals)**2

    Slices = np.zeros((numSlices, numFreqs, nChan), dtype=np.complex_)
    for iSlice in slices:
        thisSlice = data[:, ind[iSlice]:ind[iSlice] + window]
        #if sum(thisSlice)!=0:
        thisSlice = windowVals*thisSlice
        thisfft = np.fft.fft(thisSlice).T
        if frequencies is None:
            Slices[iSlice] = thisfft[0:int(window/2)]
        else:
            for fr, idx in enumerate(idx_freqs):
                Slices[iSlice, fr, :] = thisfft[idx].mean(0)

    if transpose:
        Slices = Slices.transpose(0, 2, 1)
        numFreqs = 21 #16

    if aggregate:
        Slices = np.concatenate(Slices.transpose(1,2,0), axis=0)
        Slices = np.atleast_3d(Slices).transpose(1,2,0)
        numFreqs = 1

    FFTConjSlices = np.conjugate(Slices)
    Pxx = np.divide(np.mean(abs(Slices)**2, axis=0), normVal)
    del ind, windowVals

    Cxy = []
    for fr in range(numFreqs):
        Pxy = np.dot(Slices[:, fr].T, FFTConjSlices[:, fr]) / normVal
        Pxy /= len(Slices)
        if normalize:
            Pxxx = np.outer(Pxx[fr], Pxx[fr])
            Cxy.append(abs(Pxy)**2 / Pxxx)
        else:
            Cxy.append(abs(Pxy)**2)
    return np.array(Cxy).transpose((1, 2, 0))


class Coherences(BaseEstimator, TransformerMixin):

    """Estimation of cospectral covariance matrix.

    Covariance estimation in the frequency domain. this method will return a
    4-d array with a covariance matrice estimation for each trial and in each
    frequency bin of the FFT.

    Parameters
    ----------
    window : int (default 128)
        The lengt of the FFT window used for spectral estimation.
    overlap : float (default 0.75)
        The percentage of overlap between window.
    fmin : float | None , (default None)
        the minimal frequency to be returned.
    fmax : float | None , (default None)
        The maximal frequency to be returned.
    fs : float | None, (default None)
        The sampling frequency of the signal.

    See Also
    --------
    Covariances
    HankelCovariances
    Coherences
    """

    def __init__(self, window=256, overlap=0.5, fs=200,
                 frequencies=[[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]], # , [90, 170]
                 aggregate=False, transpose=False, normalize=True):
        """Init."""
        self.window = window
        self.overlap = overlap
        self.fs = fs
        self.frequencies = frequencies
        self.aggregate = aggregate
        self.transpose = transpose
        self.normalize = normalize

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : CospCovariances instance
            The CospCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the cospectral covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of covariance matrices for each trials and for each
            frequency bin.
        """
        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            S = coherences(X[i], window=self.window, fs=self.fs,
                           overlap=self.overlap, frequencies=self.frequencies,
                           aggregate=self.aggregate, transpose=self.transpose,
                           normalize=self.normalize)
            if np.sum(S)==0:
                S = (np.zeros_like(S) + 1) * np.nan
            out.append(S)

        return np.array(out)

class PeakFreq(BaseEstimator, TransformerMixin):

    """Estimation of cospectral covariance matrix.

    Covariance estimation in the frequency domain. this method will return a
    4-d array with a covariance matrice estimation for each trial and in each
    frequency bin of the FFT.

    Parameters
    ----------
    window : int (default 128)
        The lengt of the FFT window used for spectral estimation.
    overlap : float (default 0.75)
        The percentage of overlap between window.
    fmin : float | None , (default None)
        the minimal frequency to be returned.
    fmax : float | None , (default None)
        The maximal frequency to be returned.
    fs : float | None, (default None)
        The sampling frequency of the signal.

    See Also
    --------
    Covariances
    HankelCovariances
    Coherences
    """

    def __init__(self, window=256, overlap=0.5, fs=200,
                 frequencies=[[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]]):  # , [90, 170]
        """Init."""
        self.window = window
        self.overlap = overlap
        self.fs = fs
        self.frequencies = frequencies

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : CospCovariances instance
            The CospCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the cospectral covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of covariance matrices for each trials and for each
            frequency bin.
        """

        out = []

        for x in X:
            S = peak_freq(x, window=self.window, fs=self.fs,
                          overlap=self.overlap, frequencies=self.frequencies)
            out.append(S)

        return out


class GenericTransformer(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self, method=np.mean, nanshape=(21, 1)):
        """Init."""
        self.method = method
        self.nanshape = nanshape
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            if np.isnan(x).any():
                tmp = np.ones(self.nanshape) * np.nan
            else:
                tmp = self.method(x)
            out.append(tmp)
        return np.array(out)


class BasicStats(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            m = np.mean(x, 1)
            sd = np.std(x, 1)
            ku = sp.stats.kurtosis(x, 1)
            sk = sp.stats.skew(x, 1)
            p90 = np.percentile(x, 90, axis=1)
            p10 = np.percentile(x, 10, axis=1)
            tmp = np.c_[m, sd, ku, sk, p90, p10]

            out.append(tmp)
        return np.array(out)

from pyriemann.estimation import HankelCovariances

class AutoCorrMat(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self, order=6, subsample=4):
        """Init."""
        self.order = order
        self.subsample = subsample
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        hk = HankelCovariances(delays=self.order, estimator=np.corrcoef)
        for x in X:
            tmp = []
            for a in x:
                tmp.append(hk.fit_transform(np.atleast_3d(a[::self.subsample]).transpose(0,2,1))[0])
            out.append(tmp)
        return np.array(out).transpose(0,2,3,1)

from statsmodels.tsa.ar_model import AR

class ARError(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self, order=4, subsample=4):
        """Init."""
        self.order = order
        self.subsample = subsample
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            tmp = []
            for a in x:
                ar_mod = AR(a[::self.subsample])
                ar_res = ar_mod.fit(self.order)
                bse = ar_res.bse
                if len(bse)!=(self.order + 1):
                    bse = np.array([np.nan] * (self.order + 1))
                tmp.append(bse)
            out.append(tmp)
        return np.array(out)


class VariousFeatures(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            pfd = PFD().apply(x)
            hfd = HFD().apply(x)
            hurst = Hurst().apply(x)

            tmp = np.c_[pfd, hfd, hurst]

            out.append(tmp)
        return np.array(out)

def relative_log_power(data, window=256, fs=200, overlap=0.,
                       frequencies = [[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]]): # [90, 170]
    noverlap = int(window * overlap)
    freqs, power = welch(data, fs=fs, nperseg=window, noverlap=noverlap)

    out = []
    if frequencies is None:
        out = power
    else:
        for fr in frequencies:
            tmp = (freqs >= fr[0]) & (freqs < fr[1])
            out.append((power[:, tmp].mean(1)))
    return np.log(np.array(out) / np.sum(out, 0))


def cumulative_log_power(data, window=256, fs=200, overlap=0.):
    noverlap = int(window * overlap)
    freqs, power = welch(data, fs=fs, nperseg=window, noverlap=noverlap)
    out = np.cumsum((power), 1)
    return out / np.atleast_2d(out[:, -1]).T

def spectral_edge_frequency(data, window=256, fs=200, overlap=0., edges=[0.5, 0.7, 0.8, 0.9, 0.95]):
    noverlap = int(window * overlap)
    freqs, power = welch(data, fs=fs, nperseg=window, noverlap=noverlap)
    out = np.cumsum((power), 1)
    out = out / np.atleast_2d(out[:, -1]).T
    ret = []
    if np.sum(np.isnan(out))>0:
        ret = np.ones((len(edges), 21)) * np.nan
    else:
        for edge in edges:
            tmp = []
            for ch in out:
                tmp.append(freqs[np.where(ch>edge)[0][0]])
            ret.append(tmp)
        ret = np.array(ret)
    return ret

class RelativeLogPower(BaseEstimator, TransformerMixin):

    """Relative power
    """

    def __init__(self, window=256, overlap=0.5, fs=200,
                 frequencies=[[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]]): # , [90, 170]
        """Init."""
        self.window = window
        self.overlap = overlap
        self.fs = fs
        self.frequencies = frequencies

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : CospCovariances instance
            The CospCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the cospectral covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of covariance matrices for each trials and for each
            frequency bin.
        """
        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            S = relative_log_power(X[i], window=self.window, fs=self.fs,
                           overlap=self.overlap, frequencies=self.frequencies)
            out.append(S.T)

        return np.array(out)



class CumulativeLogPower(BaseEstimator, TransformerMixin):

    """Relative power
    """

    def __init__(self, window=256, overlap=0.5, fs=200):
        """Init."""
        self.window = window
        self.overlap = overlap
        self.fs = fs

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : CospCovariances instance
            The CospCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the cospectral covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of covariance matrices for each trials and for each
            frequency bin.
        """
        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            S = cumulative_log_power(X[i], window=self.window, fs=self.fs,
                           overlap=self.overlap)
            out.append(S)

        return np.array(out)


class PowerRatios(BaseEstimator, TransformerMixin):

    """Relative power
    """

    def __init__(self, window=256, overlap=0.5, fs=200,
                 frequencies=[[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]]): # , [90, 170]
        """Init."""
        self.window = window
        self.overlap = overlap
        self.fs = fs
        self.frequencies = frequencies
        self.noverlap = int(self.window * self.overlap)

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : CospCovariances instance
            The CospCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the cospectral covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of covariance matrices for each trials and for each
            frequency bin.
        """
        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            freqs, power = welch(X[i], fs=self.fs, nperseg=self.window, noverlap=self.noverlap)

            bands = []
            for fr in self.frequencies:
                tmp = (freqs >= fr[0]) & (freqs < fr[1])
                bands.append((power[:, tmp].mean(1)))

            delta, theta, alpha, beta, gamma = bands

            alpha_delta = alpha / delta
            theta_alpha = theta / alpha
            theta_beta = theta / beta
            alpha_beta = alpha / beta
            delta_theta = delta / theta
            alpha_gamma = alpha / gamma

            S = np.array([alpha_delta, theta_alpha, theta_beta, alpha_beta, delta_theta, alpha_gamma])

            out.append(S.T)

        return np.array(out)


class SpectralEdgeFrequency(BaseEstimator, TransformerMixin):

    """Relative power
    """

    def __init__(self, window=256, overlap=0.5, fs=200, edges=[0.5, 0.7, 0.8, 0.9, 0.95]):
        """Init."""
        self.window = window
        self.overlap = overlap
        self.fs = fs
        self.edges = edges

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : CospCovariances instance
            The CospCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate the cospectral covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of covariance matrices for each trials and for each
            frequency bin.
        """
        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            S = spectral_edge_frequency(X[i], window=self.window, fs=self.fs,
                           overlap=self.overlap, edges=self.edges)
            out.append(S)

        return np.array(out)


from numpy import unwrap, angle
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin

class PLV(BaseEstimator, TransformerMixin):
    """
    Class to extracts Phase Locking Value (PLV) between pairs of channels.
    """
    def __init__(self, order=100):
        """Init."""
        self.order = order
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def plv(self, X):
        n_ch, time = X.shape
        n_pairs = int(n_ch*(n_ch-1)/2)
        # initiate matrices
        phases = np.zeros((n_ch, time))
        delta_phase_pairwise = np.zeros((n_pairs, time))
        plv = np.zeros((n_pairs,))

        # extract phases for each channel
        for c in range(n_ch):
            phases[c, :] = unwrap(angle(hilbert(X[c, :])))

        # compute phase differences
        k = 0
        for i in range(n_ch):
            for j in range(i+1, n_ch):
                delta_phase_pairwise[k, :] = phases[i, :]-phases[j, :]
                k += 1

        # compute PLV
        for k in range(n_pairs):
            plv[k] = np.abs(np.sum(np.exp(1j*delta_phase_pairwise[k, :]))/time)

        return plv

    def transform(self, X):

        out = []
        for x in X:
            tmp = self.plv(x)
            out.append(tmp)
        return np.array(out)



############## ARF FEATURES
class LineLength(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            ll = np.abs(np.diff(x))
            tmp = np.c_[ll]
            #tmp = np.c_[m, sd, ku, sk, p90, p10]

            out.append(tmp)
        return np.array(out)


class SimpleStats(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        out = []
        for x in X:
            mn = np.mean(np.abs(x), 1)
            ll = np.sum(np.abs(np.diff(x)), 1)
            sd = np.std(x, 1)
            ku = sp.stats.kurtosis(x, 1)
            sk = sp.stats.skew(x, 1)
            p90 = np.percentile(np.abs(x), 90, axis=1)

            tmp = np.c_[mn, ll, sd, ku, sk, p90]
            out.append(tmp)
        return np.array(out)




from pyrqa.settings import Settings as pyrqa_Settings
from pyrqa.time_series import TimeSeries
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from pyrqa.opencl import OpenCL
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class rqa(BaseEstimator, TransformerMixin):
    """Remove dropped packet."""

    def __init__(self, hp=0.5, lp=4, tau=1, emb_dim=3, sfreq=256):
        """Init."""
        from pyrqa.opencl import OpenCL
        self.hp = hp
        self.lp = lp
        self.tau = tau 
        self.emb_dim = emb_dim
        self.sfreq = sfreq
        pass

    def fit(self, X, y=None):
        """
        Fit, do nothing
        """
        return self

    def get_filt_eeg(self, data_arr):
        with suppress_stdout():
            n_channels = data_arr.shape[0]
            info = mne.create_info(n_channels, 
                sfreq=self.sfreq, 
                ch_types = ['eeg']*n_channels, 
                verbose=False)
            raw = mne.io.RawArray(data_arr, info)

            raw.filter(self.hp, self.lp, verbose='ERROR', fir_design='firwin')
            raw.resample(self.lp*4, npad='auto')

        return raw.get_data()
        
    def transform(self, X):
        """
        Detect and remove dropped.
        """
        opencl = OpenCL(platform_id=0, device_ids=(0,))
        out = []

        R_idx = [0, 1, 2, 3, 8, 9, 10, 14, 15, 16, 17]
        L_idx = [4, 5, 6, 7, 11, 12, 13, 18, 19, 20, 21]

        A_idx = [0, 1, 4, 5, 14, 15, 18, 19]
        P_idx = [2, 3, 6, 7, 16, 17, 20, 21]

        # startTime = datetime.now()

        for cur, x in enumerate(X):
            # RR, DET, LAM, L_max, L_entr, L_mean, and TT

            x = self.get_filt_eeg(x)

            RR = [ ]
            DET = [ ]
            LAM = [ ]
            L_max = [ ]
            L_entr = [ ]
            L_mean = [ ]
            TT = [ ]

            scaler = StandardScaler()
            scaler.fit(x)
            x = scaler.transform(x)
            
            for ch in x:
                # ch*=1000
                ch = TimeSeries(ch,
                         embedding_dimension=self.emb_dim,
                         time_delay=self.tau)
                settings = pyrqa_Settings(ch,
                        neighbourhood=FixedRadius(0.5),  #1.0
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
                computation = RQAComputation.create(settings, opencl=opencl, verbose=False)
                result = computation.run()
                
                result.min_diagonal_line_length=2
                result.min_vertical_line_length=2
                result.min_white_vertical_line_length=2

                curr_RR = result.recurrence_rate
                curr_DET = result.determinism
                curr_LAM = result.laminarity
                curr_L_max = result.longest_diagonal_line # L_max
                curr_L_entr = result.entropy_diagonal_lines # L_entr
                curr_L_mean = result.average_diagonal_line #L_mean
                curr_TT = result.trapping_time #TT

                RR.append(curr_RR)
                DET.append(curr_DET)
                LAM.append(curr_LAM)
                L_max.append(curr_L_max)
                L_entr.append(curr_L_entr)
                L_mean.append(curr_L_mean)
                TT.append(curr_TT)

            RR = np.vstack(RR)
            DET = np.vstack(DET)
            LAM = np.vstack(LAM)
            L_max = np.vstack(L_max)
            L_entr = np.vstack(L_entr)
            L_mean = np.vstack(L_mean)
            TT = np.vstack(TT)

            RR_av = np.mean(RR)
            RR_std = np.std(RR)
            DET_av = np.mean(DET)
            DET_std = np.std(DET)
            LAM_av = np.mean(LAM)
            LAM_std = np.std(LAM)
            L_max_av = np.mean(L_max)
            L_entr_av = np.mean(L_entr)
            L_mean_av = np.mean(L_mean)
            TT_av = np.mean(TT)

            tmp = np.c_[RR_av, RR_std, DET_av, DET_std, LAM_av, LAM_std, L_max_av, L_entr_av, L_mean_av, TT_av]

            out.append(tmp)
            
        # #Python 2: 
        # print datetime.now() - startTime 
        # sys.exit()
        return np.array(out)


class rqa_channel(rqa):
    """Remove dropped packet."""

    def transform(self, X):
        """
        Detect and remove dropped.
        """
        opencl = OpenCL(platform_id=0, device_ids=(0,))
        out = np.zeros([X.shape[0], X.shape[1], 7])

        for cur, x in enumerate(X):
            # RR, DET, LAM, L_max, L_entr, L_mean, and TT
            x = self.get_filt_eeg(x)

            rr = np.zeros(len(x))
            det = np.zeros(len(x))
            lam = np.zeros(len(x))
            l_max = np.zeros(len(x))
            l_entr = np.zeros(len(x))
            l_mean = np.zeros(len(x))
            tt = np.zeros(len(x))

            scaler = StandardScaler()
            scaler.fit(x)
            x = scaler.transform(x)

            for idx,ch in enumerate(x):
                # ch*=1000
                ch = TimeSeries(ch,
                         embedding_dimension=self.emb_dim,
                         time_delay=self.tau)
                settings = pyrqa_Settings(ch,
                        neighbourhood=FixedRadius(0.5),  #1.0
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
                computation = RQAComputation.create(settings, opencl=opencl, verbose=False)
                result = computation.run()
                
                result.min_diagonal_line_length=2
                result.min_vertical_line_length=2
                result.min_white_vertical_line_length=2

                rr[idx] = result.recurrence_rate
                det[idx] = result.determinism
                lam[idx] = result.laminarity
                l_max[idx] = result.longest_diagonal_line # L_max
                l_entr[idx] = result.entropy_diagonal_lines # L_entr
                l_mean[idx] = result.average_diagonal_line #L_mean
                tt[idx] = result.trapping_time #TT

            out[cur] = np.c_[rr, det, lam, l_max, l_entr, l_mean, tt]
            
        return out