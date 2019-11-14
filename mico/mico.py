"""
Parallelized Mutual Information based Feature Selection module.

Author: Daniel Homola <dani.homola@gmail.com>
License: BSD 3 clause
"""

import numpy as np
from scipy import signal
from sklearn.utils import check_X_y
from sklearn.preprocessing import StandardScaler
from psutil import cpu_count
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
import bottleneck as bn
from . import mico_utils
#from sklearn.feature_selection import VarianceThreshold
from scipy import sparse, stats
import copy
from abc import ABCMeta, abstractmethod
from colinpy import *


###############################################################################
# IO                                                                          #
###############################################################################
import errno, os, logging


def setup_logging(level, filename=None):
    logging.basicConfig(filename=filename, format='%(message)s', level=level)
    # logging.basicConfig(filename=filename, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p', level=level)


def append_message(full_msg, curr_msg, check=True, new_line=True):
    if check:
        return (full_msg + (curr_msg + "\n" if new_line else curr_msg)) if curr_msg not in full_msg else full_msg
    else:
        return full_msg + (curr_msg + "\n" if new_line else curr_msg)


def make_dir(path):
    """Private function for creating the output directory.
    @ref https://stackoverflow.com/questions/20666764/python-logging-how-to-ensure-logfile-directory-is-created
    """
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise FileNotFoundError("Failed to make directory")


###############################################################################
# Time stamp                                                                  #
###############################################################################
from time import gmtime, strftime

def get_time_stamp():
    return strftime("%Y-%m-%d-%H-%M-%S", gmtime())


###############################################################################
# MI.                                                                         #
###############################################################################
import numpy as np
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors
#from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed


def _replace_vec_nan_with_zero(vec):
    for i, e in enumerate(vec):
        if np.isnan(e):
            vec[i] = 0.0
    return vec


def _replace_mat_nan_with_zero(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i][j]):
                mat[i][j] = 0.0

    return mat


def _clean_up_MI(MI, use_nan_for_invalid_mi):
    if MI == np.inf:
        MI = 999999.0
    if MI >= 0:
        return MI
    else:
        return np.nan if use_nan_for_invalid_mi else 0.0


def get_first_mi_vector(MI_FS, k, are_data_binned):
    """
    Calculates the Mututal Information between each feature in X and y.

    This function is for when |S| = 0. We select the first feature in S.
    """
    n, p = MI_FS.X.shape
    MIs = Parallel(n_jobs=MI_FS.n_jobs)(delayed(_get_first_mi)(i, k, MI_FS, are_data_binned) for i in range(p))
    return MIs


def _get_first_mi(i, k, MI_FS, are_data_binned, use_nan_for_invalid_mi=True):
    n, p = MI_FS.X.shape
    x = MI_FS.X[:, i].reshape(n, 1)
    y = MI_FS.y.reshape(n, 1)
    #y = MI_FS.y.flatten().reshape(n, 1)

    if are_data_binned:
        if MI_FS.categorical:
            MI = mico_utils.get_mutual_information_dd(x, y)
        else:
            MI = mico_utils.get_mutual_information_cd(y, x, k)
    else:
        if MI_FS.categorical:
            MI = mico_utils.get_mutual_information_cd(x, y, k)
        else:
            MI = mico_utils.get_mutual_information_cc(x, y, k)

    # MI must be non-negative
    return _clean_up_MI(MI, use_nan_for_invalid_mi)


def get_mi_vector(MI_FS, k, F, s, are_data_binned):
    """
    Calculates the Mututal Information between each feature in F and s.

    This function is for when |S| > 1. s is the previously selected feature.
    We exploite the fact that this step is embarrassingly parallel.
    """
    MIs = Parallel(n_jobs=MI_FS.n_jobs)(delayed(_get_mi)(f, s, k, MI_FS, are_data_binned) for f in F)
    return MIs


def _get_mi(f, s, k, MI_FS, are_data_binned, use_nan_for_invalid_mi=True):
    """
    A Semidefinite Programming Based Search Strategy for Feature Selection with Mutual Information Measure
    Tofigh Naghibi , Sarah Hoffmann and Beat Pfister
    Section 2.3.1.

    :param f:
    :param s:
    :param MI_FS:
    :param are_data_binned:
    :return:
    """
    n, p = MI_FS.X.shape
    if MI_FS.method in ['JMI', 'JMIM']:
        # JMI & JMIM
        if s != f:
            # Off-diagonal.
            y = MI_FS.y.reshape(n, 1)
            #y = MI_FS.y.flatten().reshape(n, 1)
            joint = MI_FS.X[:, (s, f)]
            if are_data_binned:
                # Encoding.
                joint = mico_utils.encode_discrete_x(joint).reshape(n, 1)
                if MI_FS.categorical:
                    MI = mico_utils.get_mutual_information_dd(joint, y)
                else:
                    MI = mico_utils.get_mutual_information_cd(y, joint, k)
            else:
                if MI_FS.categorical:
                    MI = mico_utils.get_mutual_information_cd(joint, y, k)
                else:
                    MI = mico_utils.get_mutual_information_cc(joint, y, k)
        else:
            # Diagonal.
            x = MI_FS.X[:, f].reshape(n, 1)
            y = MI_FS.y.reshape(n, 1)
            #y = MI_FS.y.flatten().reshape(n)
            if are_data_binned:
                if MI_FS.categorical:
                    MI = mico_utils.get_mutual_information_dd(x, y)
                else:
                    MI = mico_utils.get_mutual_information_cd(y, x, k)
            else:
                if MI_FS.categorical:
                    MI = mico_utils.get_mutual_information_cd(x, y, k)
                else:
                    MI = mico_utils.get_mutual_information_cc(x, y, k)
    else:
        # MRMR
        if s != f:
            # Off-diagonal-- Did not use any information from y here.
            x1 = MI_FS.X[:, f].reshape(n, 1)
            x2 = MI_FS.X[:, s].reshape(n, 1)
            if are_data_binned:
                MI = mico_utils.get_mutual_information_dd(x1, x2)
            else:
                MI = mico_utils.get_mutual_information_cc(x1, x2, k)
        else:
            # Diagonal-- Did use information from y here.
            x = MI_FS.X[:, f].reshape(n, 1)
            y = MI_FS.y.reshape(n, 1)
            #y = MI_FS.y.flatten().reshape(n)
            if are_data_binned:
                if MI_FS.categorical:
                    MI = mico_utils.get_mutual_information_dd(x, y)
                else:
                    MI = mico_utils.get_mutual_information_cd(y, x, k)
            else:
                if MI_FS.categorical:
                    MI = mico_utils.get_mutual_information_cd(x, y, k)
                else:
                    MI = mico_utils.get_mutual_information_cc(x, y, k)

    # MI must be non-negative
    return _clean_up_MI(MI, use_nan_for_invalid_mi)


def get_mico_vector(MI_FS, k, F, s, offdiagonal_param, are_data_binned):
    """
    Calculates the Mututal Information between each feature in F and s.

    This function is for when |S| > 1. s is the previously selected feature.
    We exploite the fact that this step is embarrassingly parallel.
    """
    MIs = Parallel(n_jobs=MI_FS.n_jobs)(delayed(_get_mico)(f, s, k, MI_FS, offdiagonal_param, are_data_binned) for f in F)
    return MIs


def _get_mico(f, s, k, MI_FS, offdiagonal_param, are_data_binned, use_nan_for_invalid_mi=False):
    """
    A Semidefinite Programming Based Search Strategy for Feature Selection with Mutual Information Measure
    Tofigh Naghibi , Sarah Hoffmann and Beat Pfister
    Section 2.3.1.

    :param f:
    :param s:
    :param MI_FS:
    :param are_data_binned:
    :return:
    """
    n, p = MI_FS.X.shape
    clean_up = True

    if MI_FS.method in ['JMI', 'JMIM']:
        if False:
            # Three-way interaction info.
            # JMI & JMIM
            if s != f:
                # Off-diagonal -- -1/2(P-1) I(X_{.s}; X_{.k}; y).
                x1 = MI_FS.X[:, f].reshape(n, 1)
                x2 = MI_FS.X[:, s].reshape(n, 1)
                y = MI_FS.y.reshape(n, 1)
                clean_up = False # No need to clean-up as the 3-way interaction info can be negative.
                if are_data_binned:
                    if MI_FS.categorical:
                        MI = mico_utils.get_interaction_information_3way(x1, x2, y, "ddd", k) * offdiagonal_param
                    else:
                        MI = mico_utils.get_interaction_information_3way(x1, x2, y, "ddc", k) * offdiagonal_param
                else:
                    if MI_FS.categorical:
                        MI = mico_utils.get_interaction_information_3way(x1, x2, y, "ccd", k) * offdiagonal_param
                    else:
                        MI = mico_utils.get_interaction_information_3way(x1, x2, y, "ccc", k) * offdiagonal_param
            else:
                # Diagonal -- I(X_{.s}; y).
                x = MI_FS.X[:, f].reshape(n, 1)
                y = MI_FS.y.reshape(n, 1)
                #y = MI_FS.y.flatten().reshape(n)
                if are_data_binned:
                    if MI_FS.categorical:
                        MI = mico_utils.get_mutual_information_dd(x, y)
                    else:
                        MI = mico_utils.get_mutual_information_cd(y, x, k)
                        #MI = mico_utils.get_mutual_information_cd(y.reshape(n, 1), x.reshape(n), k)
                else:
                    if MI_FS.categorical:
                        MI = mico_utils.get_mutual_information_cd(x, y, k)
                    else:
                        MI = mico_utils.get_mutual_information_cc(x, y, k)
        else:
            # MI
            # JMI & JMIM
            if s != f:
                # Off-diagonal.
                y = MI_FS.y.reshape(n, 1)
                #y = MI_FS.y.flatten().reshape(n, 1)
                joint = MI_FS.X[:, (s, f)]
                if are_data_binned:
                    # Encoding.
                    joint = mico_utils.encode_discrete_x(joint).reshape(n, 1)
                    if MI_FS.categorical:
                        MI = mico_utils.get_mutual_information_dd(joint, y)
                    else:
                        MI = mico_utils.get_mutual_information_cd(y, joint, k)
                else:
                    if MI_FS.categorical:
                        MI = mico_utils.get_mutual_information_cd(joint, y, k)
                    else:
                        MI = mico_utils.get_mutual_information_cc(joint, y, k)
            else:
                # Diagonal.
                x = MI_FS.X[:, f].reshape(n, 1)
                y = MI_FS.y.reshape(n, 1)
                #y = MI_FS.y.flatten().reshape(n)
                if are_data_binned:
                    if MI_FS.categorical:
                        MI = mico_utils.get_mutual_information_dd(x, y)
                    else:
                        MI = mico_utils.get_mutual_information_cd(y, x, k)
                else:
                    if MI_FS.categorical:
                        MI = mico_utils.get_mutual_information_cd(x, y, k)
                    else:
                        MI = mico_utils.get_mutual_information_cc(x, y, k)
    else:
        # MRMR
        if s != f:
            # Off-diagonal -- Did not use any information  from y here.
            x1 = MI_FS.X[:, f].reshape(n, 1)
            x2 = MI_FS.X[:, s].reshape(n, 1)
            if are_data_binned:
                MI = mico_utils.get_mutual_information_dd(x1, x2) * offdiagonal_param
            else:
                MI = mico_utils.get_mutual_information_cc(x1, x2, k) * offdiagonal_param
        else:
            # Diagonal -- Did use information from y here.
            x = MI_FS.X[:, f].reshape(n, 1)
            y = MI_FS.y.reshape(n, 1)
            #y = MI_FS.y.flatten().reshape(n)
            if are_data_binned:
                if MI_FS.categorical:
                    MI = mico_utils.get_mutual_information_dd(x, y)
                else:
                    MI = mico_utils.get_mutual_information_cd(y, x, k)
                    #MI = mico_utils.get_mutual_information_cd(y.reshape(n, 1), x.reshape(n), k)
            else:
                if MI_FS.categorical:
                    MI = mico_utils.get_mutual_information_cd(x, y, k)
                else:
                    MI = mico_utils.get_mutual_information_cc(x, y, k)

    # MI must be non-negative
    if clean_up:
        MI = _clean_up_MI(MI, use_nan_for_invalid_mi)
        if s == f:
            MI = max(1.E-4, MI)

    return MI


class MutualInformationBase(BaseEstimator, SelectorMixin, metaclass=ABCMeta):
    """
    MI_FS stands for Mutual Information based Feature Selection.
    This class contains routines for selecting features using both
    continuous and discrete y variables. Three selection algorithms are
    implemented: JMI, JMIM and MRMR.

    This implementation tries to mimic the scikit-learn interface, so use fit,
    transform or fit_transform, to run the feature selection.

    Parameters
    ----------

    method : string, default = 'JMI'
        Which mutual information based feature selection method to use:
        - 'JMI' : Joint Mutual Information [1]
        - 'JMIM' : Joint Mutual Information Maximisation [2]
        - 'MRMR' : Max-Relevance Min-Redundancy [3]

    k : int, default = 5
        Sets the number of samples to use for the kernel density estimation
        with the kNN method. Kraskov et al. recommend a small integer between
        3 and 10.

    n_features : int or string, default = 'auto'
        If int, it sets the number of features that has to be selected from X.
        If 'auto' this is determined automatically based on the amount of
        mutual information the previously selected features share with y.

    categorical : Boolean, default = True
        If True, y is assumed to be a categorical class label. If False, y is
        treated as a continuous. Consequently this parameter determines the
        method of estimation of the MI between the predictors in X and y.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    verbose : int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays selected features
        - 2: displays selected features and mutual information

    num_bins : int, default=0
        Number of bins for binning the data.


    early_stop_steps : int, default=10
        Number of steps for early stopping.

    Attributes
    ----------

    n_features_ : int
        The number of selected features.

    support_ : array of length X.shape[1]
        The mask array of selected features.

    ranking_ : array of shape n_features
        The feature ranking of the selected features, with the first being
        the first feature selected with largest marginal MI with y, followed by
        the others with decreasing MI.

    mi_ : array of shape n_features
        The JMIM of the selected features. Usually this a monotone decreasing
        array of numbers converging to 0. One can use this to estimate the
        number of features to select. In fact this is what n_features='auto''
        tries to do heuristically.

    Examples
    --------

    import pandas as pd
    import mifs

    # load X and y
    X = pd.read_csv('my_X_table.csv', index_col=0).values
    y = pd.read_csv('my_y_vector.csv', index_col=0).values

    # define MI_FS feature selection method
    feat_selector = mifs.MutualInformationFeatureSelector()

    # find all relevant features
    feat_selector.fit(X, y)

    # check selected features
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

    References
    ----------

    [1] H. Yang and J. Moody, "Data Visualization and Feature Selection: New
        Algorithms for Nongaussian Data"
        NIPS 1999
    [2] Bennasar M., Hicks Y., Setchi R., "Feature selection using Joint Mutual
        Information Maximisation"
        Expert Systems with Applications, Vol. 42, Issue 22, Dec 2015
    [3] H. Peng, Fulmi Long, C. Ding, "Feature selection based on mutual
        information criteria of max-dependency, max-relevance,
        and min-redundancy"
        Pattern Analysis & Machine Intelligence 2005
    """

    def __init__(self,
                 method='JMI',
                 k=5,
                 n_features='auto',
                 categorical=True,
                 n_jobs=0,
                 verbose=0,
                 scale_data=True,
                 num_bins=0,
                 early_stop_steps=10):
        self.method = method
        self.k = k
        self.n_features = n_features
        self.categorical = categorical
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scale_data = scale_data
        self.num_bins = num_bins
        self._support_mask = None
        self.early_stop_steps = early_stop_steps
        # Check if n_jobs is negative
        if self.n_jobs <= 0:
            self.n_jobs = cpu_count()

        # Attributes.
        self.n_features_ = 0
        self.ranking_ = []
        self.mi_ = []

        # Logger.
        make_dir("./log/")
        instance = get_time_stamp()
        filename = "./log/{}.log".format(instance)
        #filename = None

        if self.verbose == 0:
            lv = logging.CRITICAL
        elif self.verbose == 1:
            lv = logging.INFO
        else:
            lv = logging.DEBUG
        setup_logging(level=lv, filename=filename)


    def _get_support_mask(self):
        if self._support_mask is None:
            raise ValueError('mRMR has not been fitted yet!')
        return self._support_mask

    def _scale_data(sefl, X):
        """"
        Scale each column of the data using SKL.

        Todo
        ----
        Shall we ignore binning the encoded features and discrete features?

        References
        ----------
        [1] http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

        # X must be stored in a row-wise order.
        if sparse.issparse(X):
            scalar = MaxAbsScaler()
        else:
            scalar = MinMaxScaler()  # MinMaxScaler() # StandardScaler

        return scalar.fit_transform(X)

    def _bin_data(sefl, X, num_bins):
        """"
        Bin the data using SKL.
        """
        # X must be stored in a row-wise order.
        from sklearn.preprocessing import KBinsDiscretizer

        if type(num_bins) is int:
            # bin_set = [ min(num_bins, len(array))]
            # @todo Check unique value in each feature.
            bin_set = [num_bins for _ in range(X.shape[1])]
        else:
            bin_set = num_bins

        est = KBinsDiscretizer(n_bins=bin_set, encode='ordinal', strategy='uniform').fit(X)

        return est.transform(X)

    def _are_data_binned(self):
        return self.num_bins > 0

    def _is_integer(self, x):
        return np.all(np.equal(np.mod(x, 1), 0))

    def _init_data(self, X, y):
        # checking input data and scaling it if y is continuous
        X, y = check_X_y(X, y)

        if not self.categorical:
            ss = StandardScaler()
            X = ss.fit_transform(X)
            y = ss.fit_transform(y.reshape(-1, 1))

        # Bin the data if needed.
        if self.num_bins > 0:
            print("Binning data.")
            X = self._bin_data(X, self.num_bins)
        else:
            self.num_bins = 0
        # Scale the data if needed.
        if self.scale_data:
            print("Scaling data.")
            X = self._scale_data(X)

        # sanity checks
        methods = ['JMI', 'JMIM', 'MRMR']
        if self.method not in methods:
            raise ValueError('Please choose one of the following methods:\n' +
                             '\n'.join(methods))

        if not isinstance(self.k, int):
            raise ValueError("k must be an integer.")
        if self.k < 1:
            raise ValueError('k must be larger than 0.')
        if self.categorical and np.any(self.k > np.bincount(y)):
            raise ValueError('k must be smaller than your smallest class.')

        if not isinstance(self.categorical, bool):
            raise ValueError('Categorical must be Boolean.')
        if self.categorical and np.unique(y).shape[0] > 5:
            print ('Are you sure y is categorical? It has more than 5 levels.')
        if not self.categorical and self._is_integer(y):
            print ('Are you sure y is continuous? It seems to be discrete.')
        if self._is_integer(X):
            print ('The values of X seem to be discrete. MI_FS will treat them'
                   'as continuous.')
        return X, y

    def _add_remove(self, S, F, i):
        """
        Helper function: removes ith element from F and adds it to S.
        """

        S.append(i)
        F.remove(i)
        return S, F

    def _remove(self, S, i):
        """
        Helper function: removes ith element from F and adds it to S.
        """

        S.remove(i)
        return S

    def _print_results(self, S, MIs):
        out = ''
        if self.n_features == 'auto':
            out += 'Auto selected feature #' + str(len(S)) + ' : ' + str(S[-1])
        else:
            out += ('Selected feature #' + str(len(S)) + ' / ' +
                    str(self.n_features) + ' : ' + str(S[-1]))

        if self.verbose > 1:
            out += ', ' + self.method + ' : ' + str(MIs[-1])
        print (out)


class MutualInformationForwardSelection(MutualInformationBase):
    """
    MI_FS stands for Mutual Information based Feature Selection.
    This class contains routines for selecting features using both
    continuous and discrete y variables. Three selection algorithms are
    implemented: JMI, JMIM and MRMR.

    This implementation tries to mimic the scikit-learn interface, so use fit,
    transform or fit_transform, to run the feature selection.

    Parameters
    ----------

    method : string, default = 'JMI'
        Which mutual information based feature selection method to use:
        - 'JMI' : Joint Mutual Information [1]
        - 'JMIM' : Joint Mutual Information Maximisation [2]
        - 'MRMR' : Max-Relevance Min-Redundancy [3]

    k : int, default = 5
        Sets the number of samples to use for the kernel density estimation
        with the kNN method. Kraskov et al. recommend a small integer between
        3 and 10.

    n_features : int or string, default = 'auto'
        If int, it sets the number of features that has to be selected from X.
        If 'auto' this is determined automatically based on the amount of
        mutual information the previously selected features share with y.

    categorical : Boolean, default = True
        If True, y is assumed to be a categorical class label. If False, y is
        treated as a continuous. Consequently this parameter determines the
        method of estimation of the MI between the predictors in X and y.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    verbose : int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays selected features
        - 2: displays selected features and mutual information

    num_bins : int, default=0
        Number of bins for binning the data.

    Attributes
    ----------

    n_features_ : int
        The number of selected features.

    support_ : array of length X.shape[1]
        The mask array of selected features.

    ranking_ : array of shape n_features
        The feature ranking of the selected features, with the first being
        the first feature selected with largest marginal MI with y, followed by
        the others with decreasing MI.

    mi_ : array of shape n_features
        The JMIM of the selected features. Usually this a monotone decreasing
        array of numbers converging to 0. One can use this to estimate the
        number of features to select. In fact this is what n_features='auto''
        tries to do heuristically.

    Examples
    --------

    import pandas as pd
    import mifs

    # load X and y
    X = pd.read_csv('my_X_table.csv', index_col=0).values
    y = pd.read_csv('my_y_vector.csv', index_col=0).values

    # define MI_FS feature selection method
    feat_selector = mifs.MutualInformationFeatureSelector()

    # find all relevant features
    feat_selector.fit(X, y)

    # check selected features
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

    References
    ----------

    [1] H. Yang and J. Moody, "Data Visualization and Feature Selection: New
        Algorithms for Nongaussian Data"
        NIPS 1999
    [2] Bennasar M., Hicks Y., Setchi R., "Feature selection using Joint Mutual
        Information Maximisation"
        Expert Systems with Applications, Vol. 42, Issue 22, Dec 2015
    [3] H. Peng, Fulmi Long, C. Ding, "Feature selection based on mutual
        information criteria of max-dependency, max-relevance,
        and min-redundancy"
        Pattern Analysis & Machine Intelligence 2005
    """

    def __init__(self,
                 method='JMI',
                 k=5,
                 n_features='auto',
                 categorical=True,
                 n_jobs=0,
                 verbose=0,
                 scale_data=True,
                 num_bins=0,
                 early_stop_steps=10):
        # Call base constructor.
        super(MutualInformationForwardSelection, self).__init__(
            method, k, n_features, categorical,
            n_jobs, verbose, scale_data, num_bins,
            early_stop_steps)

    def fit(self, X, y):
        """
        Fits the MI_FS feature selection with the chosen MI_FS method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        self.X, y = self._init_data(X, y)
        n, p = X.shape
        self.y = y.reshape((n, 1))

        # list of selected features
        S = []
        # list of all features
        F = list(range(p))

        # Calculate max features
        if self.n_features == 'auto':
            n_features = p
        else:
            n_features = min(p, self.n_features)

        feature_mi_matrix = np.zeros((n_features, p))
        feature_mi_matrix[:] = np.nan
        S_mi = []

        #-------------------------------------------------------------------#
        # FIND FIRST FEATURE                                                #
        #-------------------------------------------------------------------#
        xy_MI = np.array(get_first_mi_vector(self, self.k, self._are_data_binned()))
        xy_MI = _replace_vec_nan_with_zero(xy_MI)
        print("xy_MI", xy_MI)

        # choose the best, add it to S, remove it from F
        S, F = self._add_remove(S, F, bn.nanargmax(xy_MI))
        S_mi.append(bn.nanmax(xy_MI))

        # notify user
        if self.verbose > 0:
            self._print_results(S, S_mi)

        #-------------------------------------------------------------------#
        # FIND SUBSEQUENT FEATURES                                          #
        #-------------------------------------------------------------------#
        while len(S) < n_features:
            # loop through the remaining unselected features and calculate MI
            s = len(S) - 1
            feature_mi_matrix[s, F] = get_mi_vector(self, self.k, F, S[-1], self._are_data_binned())

            # make decision based on the chosen FS algorithm
            fmm = feature_mi_matrix[:len(S), F]
            if self.method == 'JMI':
                JMI = bn.nansum(fmm, axis=0)
                selected = F[bn.nanargmax(JMI)]
                S_mi.append(bn.nanmax(JMI))
            elif self.method == 'JMIM':
                JMIM = bn.nanmin(fmm, axis=0)
                if bn.allnan(JMIM):
                    break
                selected = F[bn.nanargmax(JMIM)]
                S_mi.append(bn.nanmax(JMIM))
            elif self.method == 'MRMR':
                if bn.allnan(bn.nanmean(fmm, axis=0)):
                    break
                MRMR = xy_MI[F] - bn.nanmean(fmm, axis=0)
                selected = F[bn.nanargmax(MRMR)]
                S_mi.append(bn.nanmax(MRMR))

            # record the JMIM of the newly selected feature and add it to S
            #if self.method != 'MRMR':
            #    S_mi.append(bn.nanmax(bn.nanmin(fmm, axis=0)))
            S, F = self._add_remove(S, F, selected)

            # notify user
            if self.verbose > 0:
                self._print_results(S, S_mi)

            # if n_features == 'auto', let's check the S_mi to stop
            if self.n_features == 'auto' and \
                    (self.early_stop_steps > 0 and len(S) > self.early_stop_steps):
                # smooth the 1st derivative of the MI values of previously sel
                MI_dd = signal.savgol_filter(S_mi[1:], 9, 2, 1)
                # does the mean of the last 5 converge to 0?
                if np.abs(np.mean(MI_dd[-5:])) < 1e-3:
                    break
        S = sorted(S)
        print("S: ", S)

        #-------------------------------------------------------------------#
        # SAVE RESULTS                                                      #
        #-------------------------------------------------------------------#
        self.n_features_ = len(S)
        self._support_mask = np.zeros(p, dtype=np.bool)
        self._support_mask[S] = True
        self.ranking_ = S
        self.mi_ = S_mi

        return self


class MutualInformationBackwardSelection(MutualInformationBase):
    """
    MI_FS stands for Mutual Information based Feature Selection.
    This class contains routines for selecting features using both
    continuous and discrete y variables. Three selection algorithms are
    implemented: JMI, JMIM and MRMR.

    This implementation tries to mimic the scikit-learn interface, so use fit,
    transform or fit_transform, to run the feature selection.

    Parameters
    ----------

    method : string, default = 'JMI'
        Which mutual information based feature selection method to use:
        - 'JMI' : Joint Mutual Information [1]
        - 'JMIM' : Joint Mutual Information Maximisation [2]
        - 'MRMR' : Max-Relevance Min-Redundancy [3]

    k : int, default = 5
        Sets the number of samples to use for the kernel density estimation
        with the kNN method. Kraskov et al. recommend a small integer between
        3 and 10.

    n_features : int or string, default = 'auto'
        If int, it sets the number of features that has to be selected from X.
        If 'auto' this is determined automatically based on the amount of
        mutual information the previously selected features share with y.

    categorical : Boolean, default = True
        If True, y is assumed to be a categorical class label. If False, y is
        treated as a continuous. Consequently this parameter determines the
        method of estimation of the MI between the predictors in X and y.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    verbose : int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays selected features
        - 2: displays selected features and mutual information

    num_bins : int, default=0
        Number of bins for binning the data.

    Attributes
    ----------

    n_features_ : int
        The number of selected features.

    support_ : array of length X.shape[1]
        The mask array of selected features.

    ranking_ : array of shape n_features
        The feature ranking of the selected features, with the first being
        the first feature selected with largest marginal MI with y, followed by
        the others with decreasing MI.

    mi_ : array of shape n_features
        The JMIM of the selected features. Usually this a monotone decreasing
        array of numbers converging to 0. One can use this to estimate the
        number of features to select. In fact this is what n_features='auto''
        tries to do heuristically.

    Examples
    --------

    import pandas as pd
    import mifs

    # load X and y
    X = pd.read_csv('my_X_table.csv', index_col=0).values
    y = pd.read_csv('my_y_vector.csv', index_col=0).values

    # define MI_FS feature selection method
    feat_selector = mifs.MutualInformationFeatureSelector()

    # find all relevant features
    feat_selector.fit(X, y)

    # check selected features
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

    References
    ----------

    [1] H. Yang and J. Moody, "Data Visualization and Feature Selection: New
        Algorithms for Nongaussian Data"
        NIPS 1999
    [2] Bennasar M., Hicks Y., Setchi R., "Feature selection using Joint Mutual
        Information Maximisation"
        Expert Systems with Applications, Vol. 42, Issue 22, Dec 2015
    [3] H. Peng, Fulmi Long, C. Ding, "Feature selection based on mutual
        information criteria of max-dependency, max-relevance,
        and min-redundancy"
        Pattern Analysis & Machine Intelligence 2005
    """

    def __init__(self,
                 method='JMI',
                 k=5,
                 n_features='auto',
                 categorical=True,
                 n_jobs=0,
                 verbose=0,
                 scale_data=True,
                 num_bins=0,
                 early_stop_steps=10):
        # Call base constructor.
        super(MutualInformationBackwardSelection, self).__init__(
            method, k, n_features, categorical,
            n_jobs, verbose, scale_data, num_bins,
            early_stop_steps)

    def fit(self, X, y):
        """
        Fits the MI_FS feature selection with the chosen MI_FS method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        print("BGN")
        self.X, y = self._init_data(X, y)
        n, p = X.shape
        self.y = y.reshape(n, 1)

        # list of selected features
        S = list(range(p))

        # Calculate max features.
        if self.n_features == 'auto':
            n_features = int(p * 0.2)
        else:
            n_features = min(p, self.n_features)
        print("n_features = ", n_features)

        feature_mi_matrix = np.zeros((p, p))
        feature_mi_matrix[:] = np.nan
        S_mi = []

        #-------------------------------------------------------------------#
        # CALCULATE FULL MIs                                                #
        #-------------------------------------------------------------------#
        if self.method == 'MRMR':
            xy_MI = np.array(get_first_mi_vector(self, self.k, self._are_data_binned()))
            xy_MI = _replace_vec_nan_with_zero(xy_MI)
            print("xy_MI", xy_MI)

        for s in S:
            feature_mi_matrix[s, S] = get_mi_vector(self, self.k, S, s, self._are_data_binned())
        # Ensure the MI matrix is symmetric.
        feature_mi_matrix = mico_utils.make_mat_sym(feature_mi_matrix)
        print(feature_mi_matrix)

        #-------------------------------------------------------------------#
        # FIND SUBSEQUENT FEATURES                                          #
        #-------------------------------------------------------------------#
        while len(S) >= n_features:
            fmm = np.zeros((len(S) - 1, len(S)))

            # make decision based on the chosen FS algorithm
            for i, s_remove in enumerate(S):
                S_keep = copy.deepcopy(S)
                S_keep.pop(i)
                fmm[:, i] = feature_mi_matrix[S_keep, s_remove]
                if bn.allnan(fmm[:, i]):
                    fmm[:, i] = [0 for _ in range(fmm.shape[0])]
                    print("Warning: feature [{0}] has all 0 MI.".format(i))

            if self.method == 'JMI':
                values = bn.nansum(fmm, axis=0)
                #removed_feature = S[bn.nanargmax(values)]
                removed_feature = S[bn.nanargmin(values)]
                #S_mi.append(bn.nanmax(values))
                S_mi.append(bn.nanmin(values))
            elif self.method == 'JMIM':
                values = bn.nanmin(fmm, axis=0)
                if bn.allnan(values):
                    break
                #removed_feature = S[bn.nanargmax(values)]
                removed_feature = S[bn.nanargmin(values)]
                #S_mi.append(bn.nanmax(values))
                S_mi.append(bn.nanmin(values))
            elif self.method == 'MRMR':
                if bn.allnan(bn.nanmean(fmm, axis=0)):
                    break
                values = xy_MI[S] - bn.nanmean(fmm, axis=0)
                #removed_feature = S[bn.nanargmax(values)]
                #S_mi.append(bn.nanmax(values))
                removed_feature = S[bn.nanargmin(values)]
                S_mi.append(bn.nanmin(values))
            print("values=", values)
            print("removed_feature=", removed_feature)

            # Update removed_feature feature list.
            S = self._remove(S, removed_feature)
            print(S)

            # notify user
            if self.verbose > 0:
                self._print_results(S, S_mi)

            # if n_features == 'auto', let's check the S_mi to stop
            if self.n_features == 'auto' and \
                    (self.early_stop_steps > 0 and p - len(S) > self.early_stop_steps):
                # smooth the 1st derivative of the MI values of previously sel
                MI_dd = signal.savgol_filter(S_mi[1:], 9, 2, 1)
                # does the mean of the last 5 converge to 0?
                if np.abs(np.mean(MI_dd[-5:])) < 1e-3:
                    print("Early stop.")
                    break
        S = sorted(S)

        #-------------------------------------------------------------------#
        # SAVE RESULTS                                                      #
        #-------------------------------------------------------------------#
        self.n_features_ = len(S)
        self._support_mask = np.zeros(p, dtype=np.bool)
        self._support_mask[S] = True
        self.ranking_ = S
        self.mi_ = S_mi

        return self


class MutualInformationConicOptimization(MutualInformationBase):
    """
    MI_FS stands for Mutual Information based Feature Selection.
    This class contains routines for selecting features using both
    continuous and discrete y variables. Three selection algorithms are
    implemented: JMI, JMIM and MRMR.

    This implementation tries to mimic the scikit-learn interface, so use fit,
    transform or fit_transform, to run the feature selection.

    Parameters
    ----------

    method : string, default = 'JMI'
        Which mutual information based feature selection method to use:
        - 'JMI' : Joint Mutual Information [1]
        - 'JMIM' : Joint Mutual Information Maximisation [2]
        - 'MRMR' : Max-Relevance Min-Redundancy [3]

    k : int, default = 5
        Sets the number of samples to use for the kernel density estimation
        with the kNN method. Kraskov et al. recommend a small integer between
        3 and 10.

    n_features : int or string, default = 'auto'
        If int, it sets the number of features that has to be selected from X.
        If 'auto' this is determined automatically based on the amount of
        mutual information the previously selected features share with y.

    categorical : Boolean, default = True
        If True, y is assumed to be a categorical class label. If False, y is
        treated as a continuous. Consequently this parameter determines the
        method of estimation of the MI between the predictors in X and y.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    verbose : int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays selected features
        - 2: displays selected features and mutual information

    num_bins : int, default=0
        Number of bins for binning the data.

    Attributes
    ----------

    n_features_ : int
        The number of selected features.

    support_ : array of length X.shape[1]
        The mask array of selected features.

    ranking_ : array of shape n_features
        The feature ranking of the selected features, with the first being
        the first feature selected with largest marginal MI with y, followed by
        the others with decreasing MI.

    mi_ : array of shape n_features
        The JMIM of the selected features. Usually this a monotone decreasing
        array of numbers converging to 0. One can use this to estimate the
        number of features to select. In fact this is what n_features='auto''
        tries to do heuristically.

    Examples
    --------

    import pandas as pd
    import mifs

    # load X and y
    X = pd.read_csv('my_X_table.csv', index_col=0).values
    y = pd.read_csv('my_y_vector.csv', index_col=0).values

    # define MI_FS feature selection method
    feat_selector = mifs.MutualInformationFeatureSelector()

    # find all relevant features
    feat_selector.fit(X, y)

    # check selected features
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

    References
    ----------

    [1] H. Yang and J. Moody, "Data Visualization and Feature Selection: New
        Algorithms for Nongaussian Data"
        NIPS 1999
    [2] Bennasar M., Hicks Y., Setchi R., "Feature selection using Joint Mutual
        Information Maximisation"
        Expert Systems with Applications, Vol. 42, Issue 22, Dec 2015
    [3] H. Peng, Fulmi Long, C. Ding, "Feature selection based on mutual
        information criteria of max-dependency, max-relevance,
        and min-redundancy"
        Pattern Analysis & Machine Intelligence 2005
    [4] T. Naghibi , S. Hoffmann, and B. Pfister, "A Semidefinite Programming Based Search
        Strategy for Feature Selection with Mutual Information Measure"
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(8), 2015, pp 1529--1541.
    """

    def __init__(self,
                 method='JMI',
                 k=5,
                 n_features='auto',
                 categorical=True,
                 n_jobs=0,
                 verbose=0,
                 scale_data=True,
                 num_bins=0,
                 random_state=0):
        # Call base constructor.
        super(MutualInformationConicOptimization, self).__init__(
            method, k, n_features, categorical,
            n_jobs, verbose, scale_data, num_bins)
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fits the MI_FS feature selection with the chosen MI_FS method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """

        #-------------------------------------------------------------------#
        # Initialize the parameters.                                        #
        #-------------------------------------------------------------------#
        self.X, y = self._init_data(X, y)
        n, num_features = X.shape
        self.y = y.reshape(n, 1)

        # list of features
        S = list(range(num_features))

        #-------------------------------------------------------------------#
        # Create the MI matrix.                                             #
        #-------------------------------------------------------------------#
        # Notation:
        # - N : num_features
        # - P : num_features_sel
        Q = np.zeros((num_features, num_features))
        Q[:] = 0.0

        # Calculate number of the selected features and the parameter.
        if self.n_features == 'auto':
            num_features_sel = num_features
        else:
            num_features_sel = min(num_features, self.n_features)
        offdiagonal_param = -0.5 / (num_features_sel - 1)
        print("=" * 80)
        print("num_features_sel", num_features_sel)
        print("num_features", num_features)
        print("offdiagonal_param", offdiagonal_param)
        print("S", S)

        # Calculate the MI matrix.
        for i, s in enumerate(S):
            S2 = list(range(i, num_features, 1))
            Q[s, S2] = get_mico_vector(self, self.k, S2, s, offdiagonal_param, self._are_data_binned())

        # Ensure the MI matrix is symmetric.
        for i in range(num_features):
            for j in range(i + 1, num_features):
                Q[j, i] = Q[i, j]
        #feature_mi_matrix = mico_utils.make_mat_sym(feature_mi_matrix)

        print(Q)

        # Calculate Q^T e.
        QTe = np.zeros(num_features)
        for s in S:
            QTe[s] = Q[s, :].sum()
            assert(QTe[s] == Q[:, s].sum())

        # Create Q^u matrix.
        # The first row/column is the auxiliary variable.
        Qu = np.zeros((num_features + 1, num_features + 1))
        Qu[0, 0] = 0.0
        Qu[1:(num_features + 1), 1:(num_features + 1)] = Q
        Qu[0, 1:(num_features + 1)] = QTe
        Qu[1:(num_features + 1), 0] = QTe

        print("=" * 80)
        print(Qu)

        #-------------------------------------------------------------------#
        # Input optimization model.                                         #
        #                                                                   #
        # See                                                               #
        #   A Semidefinite Programming Based Search Strategy for Feature    #
        #   Selection with Mutual Information Measure, Eq (26)              #
        #-------------------------------------------------------------------#
        def map_ij_to_k(i, j, size):
            return i * size + j

        def map_k_to_ij(k, size):
            return int(k/size), int(k%size)

        # Cone information.
        semidefinite_cone = list(range((num_features + 1) * (num_features + 1)))
        #print("semidefinite_cone", semidefinite_cone)

        # Constraint 1.
        row1_idx = []
        row1_val = []
        for i in range(1, num_features + 1):
            for j in range(1, num_features + 1):
                row1_idx += [map_ij_to_k(i, j, num_features + 1)]
                row1_val += [1.0]
        row1_rhs = (2.0 * num_features_sel - num_features) ** 2
        #print("row1_idx", row1_idx)
        #print("row1_val", row1_val)
        #print("row1_rhs", row1_rhs)

        # Constraint 2.
        row2_idx = []
        row2_val = []
        for i in range(1, num_features + 1):
            row2_idx += [map_ij_to_k(0, i, num_features + 1)]
            row2_val += [1.0]
        for i in range(1, num_features + 1):
            row2_idx += [map_ij_to_k(i, 0, num_features + 1)]
            row2_val += [1.0]
        row2_rhs = 2.0 * (2.0 * num_features_sel - num_features)
        #print("row2_idx", row2_idx)
        #print("row2_val", row2_val)
        #print("row2_rhs", row2_rhs)

        # Model.
        CLN_INFINITY = ClnModel.get_infinity()
        model = ClnModel()

        try:
            # Step 1. Create a model and change the parameters.
            # Create an empty model.
            model.create_mdl()

            # Set parameters.
            model.set_ipa("Model/Verbose", 3 if self.verbose >= 2 else self.verbose)
            #model.set_ipa("Model/Presolver/PresolveLv", 1)
            model.set_ipa("Ips/Solver/NumThreads", self.n_jobs)
            model.set_ipa("Ips/Solver/Type", 2)

            # Step 2. Input model.
            # Change to maximization problem.
            model.set_max_obj_sense()

            # Add variables.
            # One semidefinite block.
            for i in range(0, num_features + 1):
                for j in range(0, num_features + 1):
                    model.add_col(-CLN_INFINITY, CLN_INFINITY, Qu[i, j], [], [])

            # Add constraints.
            # - Cnstraint 1:
            #   \sum_{i,j}^N Y_{ij} = (2P - N)^2.
            model.add_row(row1_rhs, row1_rhs, row1_idx, row1_val)

            # - Constraint 2:
            #   \sum_{i}^N Y_{i0} + \sum_{i}^N Y_{0i} = 2 * (2P - N).
            model.add_row(row2_rhs, row2_rhs, row2_idx, row2_val)

            # - Constraint 3:
            #   diag(Y) = e.
            for i in range(0, num_features + 1):
                model.add_row(1.0, 1.0, [map_ij_to_k(i, i, num_features + 1)], [1.0])

            # - Add semidefinite conic constraint.
            model.add_semi_definite_cone(semidefinite_cone)

            # Step 3. Solve the problem and populate the result.
            # Solve the problem.
            model.solve_prob()

        except ClnError as e:
            logging.error("Received Colin exception.")
            logging.error(" - Explanation   : {}".format(e.message))
            logging.error(" - Code          : {}".format(e.code))
        except Exception as e:
            logging.error("Received exception.")
            logging.error(" - Explanation   : {}".format(e))
        finally:
            # Step 4. Display the result and free the model.
            if self.verbose >= 1:
                model.display_results()

            # Retrieve result code, optimization status, and solution status.
            result = int(model.get_solver_result())
            opt_status = model.get_solver_opt_status()
            sol_status = model.get_solver_sol_status()
            has_solution = model.has_solution()

            # Note: Non-optimal status (ENM_OPT_STATUS_OPTIMAL).
            make_dir("./log/")
            if opt_status != 1:
                if has_solution:
                    # TODO Support SDP in LP/MPS output.
                    # Can continue.
                    logging.warning("Colin failed to solve the problem to optimality, but at least a primal solution is available.")
                    instance = get_time_stamp()
                    filename = "./log/warn_code_{0}_time_{1}.lp".format(opt_status, instance)
                    #model.write_prob(filename)
                else:
                    # Cannot continue.
                    instance = get_time_stamp()
                    filename = "./log/err_code_{0}_time_{1}.lp".format(opt_status, instance)
                    #model.write_prob(filename)

                    msg = "Colin failed to solve the problem to optimality. Terminated."
                    logging.error(msg)
                    raise ValueError(msg)

                logging.info(" - Result        : {0} ({1})".format(model.explain_result(result), result))
                logging.info(" - Opt status    : {0} ({1})".format(model.explain_opt_status(opt_status), opt_status))
                logging.info(" - Sol status    : {0} ({1})".format(model.explain_sol_status(sol_status), sol_status))
                logging.info(" - Has solution  : {0}".format(has_solution))
            else:
                logging.info("Colin solved the problem to optimality.")

            if has_solution:
                prim_soln = model.get_prim_soln()

            model.free_mdl()

        # Create covariance matrix.
        mean_vec = np.zeros(num_features + 1)
        cov_mat = []
        for i in range(0, num_features + 1):
            for j in range(0, num_features + 1):
                cov_mat.append(prim_soln[map_ij_to_k(i, j, num_features + 1)])
        #cov_mat = list(map(lambda x : round(x, 4), cov_mat))
        #print(cov_mat)

        # Perturbation.
        for k in range(0, (num_features + 1) * (num_features + 1)):
            i, j = map_k_to_ij(k, num_features + 1)
            if i == j:
                # Add eps to diagonal.
                pass
                cov_mat[k] += 1.E-4

        cov_mat = np.matrix(cov_mat).reshape((num_features + 1), (num_features + 1))
        #print("cov_mat", cov_mat)

        pdf = stats.multivariate_normal(mean_vec, cov_mat)
        for i in range(10):
            sampled_pt = pdf.rvs(1, random_state=i)
            ref_pt = sampled_pt[0]
            print("ref_pt", ref_pt)
            if abs(ref_pt) < 0.1:
                continue
            if ref_pt >= 0:
                S = [i for i, e in enumerate(sampled_pt[1:len(sampled_pt)]) if e >= 0]
            else:
                S = [i for i, e in enumerate(sampled_pt[1:len(sampled_pt)]) if e <= 0]
            print("S: ", S)

        print("Sel features =", len(S))

        #-------------------------------------------------------------------#
        # SAVE RESULTS                                                      #
        #-------------------------------------------------------------------#
        self.n_features_ = len(S)
        self._support_mask = np.zeros(num_features, dtype=np.bool)
        self._support_mask[S] = True
        self.ranking_ = S
        self.mi_ = []

        return self
