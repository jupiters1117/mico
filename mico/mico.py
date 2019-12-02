"""
This module contains a scikit-learn compatible implementation of mutual information based feature selection methods.
"""

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
from scipy.stats import rankdata
import copy
from abc import ABCMeta, abstractmethod
import math

DEBUG = False


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


def get_first_mi_vector(MI_FS, k, are_data_binned, n_jobs=1):
    """
    Calculates the Mututal Information between each feature in X and y.

    This function is for when |S| = 0. We select the first feature in S.
    """
    n, p = MI_FS.X.shape
    if n_jobs <= 1:
        MIs = [_get_first_mi(i, k, MI_FS, are_data_binned) for i in range(p)]
    else:
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


def get_mi_vector(MI_FS, k, F, s, are_data_binned, n_jobs=1):
    """
    Calculates the Mututal Information between each feature in F and s.

    This function is for when |S| > 1. s is the previously selected feature.
    We exploit the fact that this step is embarrassingly parallel.
    """
    if n_jobs <= 1:
        MIs = [_get_mi(f, s, k, MI_FS, are_data_binned) for f in F]
    else:
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


def get_mico_vector(MI_FS, k, F, s, offdiagonal_param, are_data_binned, Nx=None, n_jobs=1):
    """
    Calculates the Mututal Information between each feature in F and s.

    This function is for when |S| > 1. s is the previously selected feature.
    We exploit the fact that this step is embarrassingly parallel.
    """
    #print("Start {}".format(s))
    if n_jobs <= 1:
        MIs = [_get_mico(f, s, k, MI_FS, offdiagonal_param, are_data_binned, Nx=Nx) for f in F]
    else:
        MIs = Parallel(n_jobs=MI_FS.n_jobs)(delayed(_get_mico)(f, s, k, MI_FS, offdiagonal_param, are_data_binned, Nx=Nx) for f in F)
    #print("Done {}".format(s))
    return MIs


def _get_mico(f, s, k, MI_FS, offdiagonal_param, are_data_binned, use_nan_for_invalid_mi=False, Nx=None):
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
                        MI = mico_utils.get_mutual_information_cd(x, y, k, Nx)
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
                        MI = mico_utils.get_mutual_information_cd(joint, y, k, Nx)
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
                        MI = mico_utils.get_mutual_information_cd(x, y, k, Nx)
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
                    MI = mico_utils.get_mutual_information_cd(y, x, k, Nx)
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
    Base class for feature selection implementation.

    This base class contains routines for selecting features implementation.


    Parameters
    ----------

    method : string, optional (default='JMI')

        A string that specifies the mutual information measure. Possible options are:

        - **JMI** : Joint Mutual Information [1]_
        - **JMIM** : Joint Mutual Information Maximisation [2]_
        - **MRMR** : Max-Relevance Min-Redundancy [3]_

    k : int, optional (default=5)

        An integer to specify number of samples for the kernel density estimation using KNN. A small integer between 3 and 10 is recommended. Note that this parameter is applicable only if `num_bins` is set to 0.

    n_features : int or string, optional (default='auto')

        A positive integer number that specifies the number of features to be selected.
        If 'auto' is used, then this number will be determined automatically.

    categorical : boolean, optional (default=True)

        If `categorical` is `True`, then the dependent variable is assumed to be a categorical class label. Otherwise, the dependent variable is assumed to be a continuous variable.

    n_jobs : int, optional (default=1)

        The number of threads allowed for the computations. If `n_jobs` :math:`<= 0`, then it will be determined internally.

    verbose : int, optional (default=0)

        An integer number to specify the verbosity level. Available options are:

        - 0: Disable output.
        - 1: Display summary results.
        - 2: Displays all results.

    scale_data : boolean, optional (default=True)

        An boolean flag that specifies if the input data input shall be scaled first.

    num_bins : int, optional (default=0)

        An integer number to specify the number of bins allowed for data binning (bucketing). If :math:`0`, then data binning will be disabled, and KNN will be used for kernal density estimation.


    References
    ----------

    .. [1] H Yang and J Moody, "Data Visualization and Feature Selection: New Algorithms for Nongaussian Data", NIPS 1999. [`Pre-print <https://papers.nips.cc/paper/1779-data-visualization-and-feature-selection-new-algorithms-for-nongaussian-data.pdf>`_]
    .. [2] M Bennasar, Y Hicks, abd R Setchi, "Feature selection using Joint Mutual Information Maximisation", Expert Systems with Applications, 42(22), pp. 8520--8532, 2015 [`pre-print <https://core.ac.uk/download/pdf/82448198.pdf>`_]
    .. [3] H Peng, F Long, and C Ding, "Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy", IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), pp. 1226--1238, 2005. [`Pre-print <http://ranger.uta.edu/~chqding/papers/mRMR_PAMI.pdf>`_]
    """
    def __init__(self,
                 method='JMI',
                 k=5,
                 n_features='auto',
                 categorical=True,
                 n_jobs=0,
                 verbose=0,
                 scale_data=True,
                 num_bins=0):
        self.method = method
        self.k = k
        self.n_features = n_features
        self.categorical = categorical
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scale_data = scale_data
        self.num_bins = num_bins
        self._support_mask = None
        # Check if n_jobs is negative
        if self.n_jobs <= 0:
            self.n_jobs = cpu_count()

        # Logger.
        #make_dir("./log/")
        #instance = get_time_stamp()
        #filename = "./log/{}.log".format(instance)
        filename = None

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
            logging.info("Started binning data.")
            X = self._bin_data(X, self.num_bins)
        else:
            self.num_bins = 0
        # Scale the data if needed.
        if self.scale_data:
            logging.info("Started scaling data.")
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
            logging.warning('Are you sure y is categorical? It has more than 5 levels.')
        if not self.categorical and self._is_integer(y):
            logging.warning('Are you sure y is continuous? It seems to be discrete.')
        if self._is_integer(X):
            logging.warning('The values of X seem to be discrete. MI_FS will treat them as continuous.')

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

    def _calc_xQx(self, Q, curr_soln):
        score = 0
        n = Q.shape[0]
        for i in range(n):
            if curr_soln[i] > 0:
                for j in range(n):
                    if curr_soln[j] > 0 and not np.isnan(Q[i, j]):
                        score += Q[i, j]

        return score

    def _convert_idx_array_to_mask_array(self, idx_array, n):
        mask_array = np.zeros(n, dtype=int)
        mask_array[idx_array] = 1
        return mask_array

    def _convert_mask_array_to_idx_array(self, mask_array):
        return np.where(mask_array)

    def _generate_feature_importances(self, best_soln, Q, num_features):
        feature_importances = []
        require_clean_run = False
        best_soln_binary = self._convert_idx_array_to_mask_array(best_soln, num_features)
        best_score = self._calc_xQx(Q, best_soln_binary)

        for i in range(len(best_soln_binary)):
            if best_soln_binary[i] == 1:
                # Remove the feature and recalculate the score.
                best_soln_binary[i] = 0
                curr_score = self._calc_xQx(Q, best_soln_binary)
                diff_score = best_score - curr_score
                best_soln_binary[i] = 1
                if diff_score < 0:
                    logging.info("Warning: Feature {} has negative score {1}".format(i, diff_score))
                    best_soln_binary[i] = 0
                    require_clean_run = True
                feature_importances.append(diff_score)
            else:
                feature_importances.append(0.0)

        if require_clean_run:
            best_soln = self._convert_mask_array_to_idx_array(best_soln_binary)
            best_score = self._calc_xQx(Q, best_soln_binary)

        # Normalize and rank.
        if sum(feature_importances) <= 0:
            raise ValueError("Failed: all features have zero importance score. Try using a smaller value for parameter k.")
        feature_importances = feature_importances / sum(feature_importances)
        feature_raking = list(map(int, rankdata(-feature_importances)))

        return best_soln, best_score, feature_importances, feature_raking


class MutualInformationForwardSelection(MutualInformationBase):
    """
    Forward selection approach for feature selection.

    This class implements a forward selection approach for feature selection with mutual information (MI) measure.


    Parameters
    ----------

    method : string, optional (default='JMI')

        A string that specifies the mutual information measure. Possible options are:

        - **JMI** : Joint Mutual Information [1]_
        - **JMIM** : Joint Mutual Information Maximisation [2]_
        - **MRMR** : Max-Relevance Min-Redundancy [3]_

    k : int, optional (default=5)

        An integer to specify number of samples for the kernel density estimation using KNN. A small integer between 3 and 10 is recommended. Note that this parameter is applicable only if `num_bins` is set to 0.

    n_features : int or string, optional (default='auto')

        A positive integer number that specifies the number of features to be selected.
        If 'auto' is used, then this number will be determined automatically.

    categorical : boolean, optional (default=True)

        If `categorical` is `True`, then the dependent variable is assumed to be a categorical class label. Otherwise, the dependent variable is assumed to be a continuous variable.

    n_jobs : int, optional (default=1)

        The number of threads allowed for the computations. If `n_jobs` :math:`<= 0`, then it will be determined internally.

    verbose : int, optional (default=0)

        An integer number to specify the verbosity level. Available options are:

        - 0: Disable output.
        - 1: Display summary results.
        - 2: Displays all results.

    scale_data : boolean, optional (default=True)

        An boolean flag that specifies if the input data input shall be scaled first.

    num_bins : int, optional (default=0)

        An integer number to specify the number of bins allowed for data binning (bucketing). If :math:`0`, then data binning will be disabled, and KNN will be used for kernal density estimation.

    early_stop_steps : int, optional (default=10)

         An positive integer to specify the iteration limit allowed without improvement on the MI measure.


    Attributes
    ----------

    n_features_ : int

        The number of selected features.

    feature_importances_ : array of shape n_features

         The feature importance scores of the selected features.

    ranking_ : array of shape n_features

        The feature ranking of the selected features, with the first being the first feature selected with largest marginal MI with y, followed by
        the others with decreasing MI.

    mi_ : array of shape n_features

        The MI measure of the selected features. Usually this a monotone decreasing array of numbers converging to 0. One can use this to estimate the
        number of features to select. In fact this is what n_features='auto' tries to do heuristically.


    Examples
    --------

    This implementation follows the scikit-learn API convention:

     - ``fit(X, y)``
     - ``transform(X)``
     - ``fit_transform(X, y)``


    References
    ----------

    .. [1] H Yang and J Moody, "Data Visualization and Feature Selection: New Algorithms for Nongaussian Data", NIPS 1999. [`Pre-print <https://papers.nips.cc/paper/1779-data-visualization-and-feature-selection-new-algorithms-for-nongaussian-data.pdf>`_]
    .. [2] M Bennasar, Y Hicks, abd R Setchi, "Feature selection using Joint Mutual Information Maximisation", Expert Systems with Applications, 42(22), pp. 8520--8532, 2015 [`pre-print <https://core.ac.uk/download/pdf/82448198.pdf>`_]
    .. [3] H Peng, F Long, and C Ding, "Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy", IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), pp. 1226--1238, 2005. [`Pre-print <http://ranger.uta.edu/~chqding/papers/mRMR_PAMI.pdf>`_]
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
            n_jobs, verbose, scale_data, num_bins)
        self.early_stop_steps = early_stop_steps

        # Attributes.
        self.n_features_ = 0
        self.feature_importances_ = []
        self.ranking_ = []
        self.mi_ = []

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
        num_features = p
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

        # Populate information.
        self._print_init_result()

        #-------------------------------------------------------------------#
        # Find first feature                                                #
        #-------------------------------------------------------------------#
        xy_MI = np.array(get_first_mi_vector(self, self.k, self._are_data_binned(), n_jobs=1))
        xy_MI = _replace_vec_nan_with_zero(xy_MI)
        if sum(xy_MI) <= 0:
            logging.warning("Warning: The first MI are all zeros. Try using a different value of k.")

        # choose the best, add it to S, remove it from F
        selected = bn.nanargmax(xy_MI)
        S, F = self._add_remove(S, F, selected)
        S_mi.append(selected)

        # Populate information.
        self._print_curr_result(S, selected, S_mi)
        self.ranking_.append(selected)

        #-------------------------------------------------------------------#
        # Find Subsequent features                                          #
        #-------------------------------------------------------------------#
        while len(S) < n_features:
            # loop through the remaining unselected features and calculate MI
            s = len(S) - 1
            feature_mi_matrix[s, F] = get_mi_vector(self, self.k, F, S[-1], self._are_data_binned(), n_jobs=1)

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

            # record the newly selected feature and add it to S.
            S, F = self._add_remove(S, F, selected)
            self.ranking_.append(selected)

            # Populate information.
            self._print_curr_result(S, selected, S_mi)

            # Perform early stops.
            if self.n_features == 'auto' and \
                    (self.early_stop_steps > 0 and len(S) > self.early_stop_steps):
                # smooth the 1st derivative of the MI values of previously sel
                MI_dd = signal.savgol_filter(S_mi[1:], 9, 2, 1)
                # does the mean of the last 5 converge to 0?
                if np.abs(np.mean(MI_dd[-5:])) < 1e-3:
                    break
        S = sorted(S)

        #-------------------------------------------------------------------#
        # Calculate final MIs                                               #
        #-------------------------------------------------------------------#
        logging.info("Started calculating final MI matrix.")
        num_features_S = len(S)
        feature_mi_matrix = np.zeros((p, p))
        feature_mi_matrix[:] = np.nan
        if True:
            # Parallelize the outer loop - faster.
            feature_mi_matrix_partial_tri = Parallel(n_jobs=self.n_jobs)(delayed(get_mi_vector)(
                self, self.k, S[i:], s, self._are_data_binned(), n_jobs=1) for i, s in enumerate(S))
            # Get the MI matrix.
            for i in range(num_features_S):
                for j in range(i, num_features_S):
                    feature_mi_matrix[S[i], S[j]] = feature_mi_matrix_partial_tri[i][j - i]
                    if i != j:
                        feature_mi_matrix[S[j], S[i]] = feature_mi_matrix[S[i], S[j]]
        else:
            # Parallelize the inner loop - slower.
            all_S = list(range(p))
            for s in all_S:
                feature_mi_matrix[s, all_S] = get_mi_vector(self, self.k, all_S, s, self._are_data_binned(), n_jobs=self.n_jobs)
            # Ensure the MI matrix is symmetric.
            feature_mi_matrix = mico_utils.make_mat_sym(feature_mi_matrix)

        #-------------------------------------------------------------------#
        # Generate feature importance scores.                               #
        #-------------------------------------------------------------------#
        S, best_score, feature_importances, feature_raking = \
            self._generate_feature_importances(best_soln=S, Q=feature_mi_matrix, num_features=num_features)

        #-------------------------------------------------------------------#
        # Save results                                                      #
        #-------------------------------------------------------------------#
        num_features_sel = n_features
        self.n_features_ = len(S)
        self.feature_importances_ = feature_importances
        self._support_mask = np.zeros(p, dtype=np.bool)
        self._support_mask[S] = True
        self.mi_ = S_mi

        self._print_end_result(num_features_sel=num_features_sel, num_features=num_features, best_soln=S)

        return self

    def _print_init_result(self):
        logging.info("Started MIFS.")
        logging.info(" - Method        : {}".format(self.method))
        logging.info(" - Num. threads  : {}".format(self.n_jobs))
        logging.info(" - Num. features : {}".format(self.n_features))
        logging.info("{0:>5}{1:>5}{2:15}".format("Iter", "Sel", "    Current MI"))

    def _print_curr_result(self, S, selected, MIs):
        logging.info("{0:>5}{1:>5}{2:+14.6E}".format(len(S), selected, MIs[-1]))

    def _print_end_result(self, num_features_sel, num_features, best_soln):
        logging.info("Done MIFS.")
        logging.info(" - Total feat.   : {}".format(num_features))
        logging.info(" - Target feat.  : {}".format(num_features_sel))
        logging.info(" - Actual feat.  : {}".format(len(best_soln)))


class MutualInformationBackwardElimination(MutualInformationBase):
    """
    Backward elimination approach for feature selection.

    This class implements a backward selection approach for feature selection with mutual information (MI) measure.


    Parameters
    ----------

    method : string, optional (default='JMI')

        A string that specifies the mutual information measure. Possible options are:

        - **JMI** : Joint Mutual Information [1]_
        - **JMIM** : Joint Mutual Information Maximisation [2]_
        - **MRMR** : Max-Relevance Min-Redundancy [3]_

    k : int, optional (default=5)

        An integer to specify number of samples for the kernel density estimation using KNN. A small integer between 3 and 10 is recommended. Note that this parameter is applicable only if `num_bins` is set to 0.

    n_features : int or string, optional (default='auto')

        A positive integer number that specifies the number of features to be selected.
        If 'auto' is used, then this number will be determined automatically.

    categorical : boolean, optional (default=True)

        If `categorical` is `True`, then the dependent variable is assumed to be a categorical class label. Otherwise, the dependent variable is assumed to be a continuous variable.

    n_jobs : int, optional (default=1)

        The number of threads allowed for the computations. If `n_jobs` :math:`<= 0`, then it will be determined internally.

    verbose : int, optional (default=0)

        An integer number to specify the verbosity level. Available options are:

        - 0: Disable output.
        - 1: Display summary results.
        - 2: Displays all results.

    scale_data : boolean, optional (default=True)

        An boolean flag that specifies if the input data input shall be scaled first.

    num_bins : int, optional (default=0)

        An integer number to specify the number of bins allowed for data binning (bucketing). If :math:`0`, then data binning will be disabled, and KNN will be used for kernal density estimation.


    Attributes
    ----------

    n_features_ : int

        The number of selected features.

    feature_importances_ : array of shape n_features

         The feature importance scores of the selected features.

    ranking_ : array of shape n_features

        The feature ranking of the selected features, with the first being the first feature selected with largest marginal MI with y, followed by
        the others with decreasing MI.

    mi_ : array of shape n_features

        The MI measure of the selected features. Usually this a monotone decreasing array of numbers converging to 0. One can use this to estimate the
        number of features to select. In fact this is what n_features='auto' tries to do heuristically.


    Examples
    --------

    This implementation follows the scikit-learn API convention:

     - ``fit(X, y)``
     - ``transform(X)``
     - ``fit_transform(X, y)``


    References
    ----------

    .. [1] H Yang and J Moody, "Data Visualization and Feature Selection: New Algorithms for Nongaussian Data", NIPS 1999. [`Pre-print <https://papers.nips.cc/paper/1779-data-visualization-and-feature-selection-new-algorithms-for-nongaussian-data.pdf>`_]
    .. [2] M Bennasar, Y Hicks, abd R Setchi, "Feature selection using Joint Mutual Information Maximisation", Expert Systems with Applications, 42(22), pp. 8520--8532, 2015 [`pre-print <https://core.ac.uk/download/pdf/82448198.pdf>`_]
    .. [3] H Peng, F Long, and C Ding, "Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy", IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), pp. 1226--1238, 2005. [`Pre-print <http://ranger.uta.edu/~chqding/papers/mRMR_PAMI.pdf>`_]
    """
    def __init__(self,
                 method='JMI',
                 k=5,
                 n_features='auto',
                 categorical=True,
                 n_jobs=0,
                 verbose=0,
                 scale_data=True,
                 num_bins=0):
        # Call base constructor.
        super(MutualInformationBackwardElimination, self).__init__(
            method, k, n_features, categorical,
            n_jobs, verbose, scale_data, num_bins)

        # Deprecated.
        self.early_stop_steps = 0

        # Attributes.
        self.n_features_ = 0
        self.feature_importances_ = []

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
        num_features = p
        self.y = y.reshape(n, 1)

        # list of selected features
        S = list(range(p))

        # Calculate max features.
        if self.n_features == 'auto':
            n_features = int(p * 0.2)
        else:
            n_features = min(p, self.n_features)

        feature_mi_matrix = np.zeros((p, p))
        feature_mi_matrix[:] = np.nan
        S_mi = []
        # Populate information.
        self._print_init_result()

        #-------------------------------------------------------------------#
        # Calculate full MIs                                                #
        #-------------------------------------------------------------------#
        logging.info("Started calculating full MI matrix.")
        if self.method == 'MRMR':
            xy_MI = np.array(get_first_mi_vector(self, self.k, self._are_data_binned(), n_jobs=1))
            xy_MI = _replace_vec_nan_with_zero(xy_MI)

        if True:
            # Parallelize the outer loop - faster.
            feature_mi_matrix_tri = Parallel(n_jobs=self.n_jobs)(delayed(get_mi_vector)(
                self, self.k, list(range(i, num_features, 1)), s, self._are_data_binned(), n_jobs=1) for i, s in enumerate(S))
            for i, s in enumerate(S):
                feature_mi_matrix[s, list(range(i, num_features, 1))] = feature_mi_matrix_tri[s]
            # Ensure the MI matrix is symmetric.
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    feature_mi_matrix[j, i] = feature_mi_matrix[i, j]
        else:
            # Parallelize the inner loop - slower.
            for s in S:
                feature_mi_matrix[s, S] = get_mi_vector(self, self.k, S, s, self._are_data_binned(), n_jobs=self.n_jobs)
            # Ensure the MI matrix is symmetric.
            feature_mi_matrix = mico_utils.make_mat_sym(feature_mi_matrix)

        #-------------------------------------------------------------------#
        # Find subsequent features                                          #
        #-------------------------------------------------------------------#
        logging.info("Started backward selection.")
        while len(S) >= n_features + 1:
            fmm = np.zeros((len(S) - 1, len(S)))

            # make decision based on the chosen FS algorithm
            for i, s_remove in enumerate(S):
                S_keep = copy.deepcopy(S)
                S_keep.pop(i)
                fmm[:, i] = feature_mi_matrix[S_keep, s_remove]
                if bn.allnan(fmm[:, i]):
                    fmm[:, i] = [0 for _ in range(fmm.shape[0])]
                    logging.warning("Feature [{0}] has all 0 MI.".format(i))

            if self.method == 'JMI':
                values = bn.nansum(fmm, axis=0)
                removed_feature = S[bn.nanargmin(values)]
                S_mi.append(bn.nanmin(values))
            elif self.method == 'JMIM':
                values = bn.nanmin(fmm, axis=0)
                if bn.allnan(values):
                    break
                removed_feature = S[bn.nanargmin(values)]
                S_mi.append(bn.nanmin(values))
            elif self.method == 'MRMR':
                if bn.allnan(bn.nanmean(fmm, axis=0)):
                    break
                values = xy_MI[S] - bn.nanmean(fmm, axis=0)
                removed_feature = S[bn.nanargmin(values)]
                S_mi.append(bn.nanmin(values))

            # Update removed_feature feature list.
            S = self._remove(S, removed_feature)

            # Populate information.
            self._print_curr_result(S, removed_feature, S_mi)

            # if n_features == 'auto', let's check the S_mi to stop
            # Deprecated.
            if False and self.n_features == 'auto' and \
                    (self.early_stop_steps > 0 and p - len(S) > self.early_stop_steps):
                # smooth the 1st derivative of the MI values of previously sel
                MI_dd = signal.savgol_filter(S_mi[1:], 9, 2, 1)
                # does the mean of the last 5 converge to 0?
                if np.abs(np.mean(MI_dd[-5:])) < 1e-3:
                    logging.info("Early stopping criteria reached.")
                    break
        S = sorted(S)

        #-------------------------------------------------------------------#
        # Generate feature importance scores.                               #
        #-------------------------------------------------------------------#
        S, best_score, feature_importances, feature_raking = \
            self._generate_feature_importances(best_soln=S, Q=feature_mi_matrix, num_features=num_features)

        #-------------------------------------------------------------------#
        # Save results                                                      #
        #-------------------------------------------------------------------#
        num_features_sel = n_features
        self.n_features_ = len(S)
        self.feature_importances_ = feature_importances
        self._support_mask = np.zeros(p, dtype=np.bool)
        self._support_mask[S] = True
        #self.mi_ = S_mi

        self._print_end_result(num_features_sel=num_features_sel, num_features=num_features, best_soln=S)

        return self

    def _print_init_result(self):
        logging.info("Started MIBS.")
        logging.info(" - Method        : {}".format(self.method))
        logging.info(" - Num. threads  : {}".format(self.n_jobs))
        logging.info(" - Num. features : {}".format(self.n_features))
        logging.info("{0:>5}{1:>5}{2:15}".format("Iter", "Rmv", "    Current MI"))

    def _print_curr_result(self, S, selected, MIs):
        logging.info("{0:>5}{1:>5}{2:+14.6E}".format(len(S), selected, MIs[-1]))

    def _print_end_result(self, num_features_sel, num_features, best_soln):
        logging.info("Done MIBS.")
        logging.info(" - Total feat.   : {}".format(num_features))
        logging.info(" - Target feat.  : {}".format(num_features_sel))
        logging.info(" - Actual feat.  : {}".format(len(best_soln)))


class MutualInformationConicOptimization(MutualInformationBase):
    """
    Conic optimization approach for feature selection.

    This class implements a conic optimization based feature selection method with mutual information (MI) measure [1]_. This idea is to formulate the selection problem as a pure-binary quadratic optimization problem, which can be heuristically solved by an efficient randomization algorithm via semidefinite programming [2]_. Optimization software **Colin** [6]_ is used for solving the underlying conic optimization problems.


    Parameters
    ----------

    method : string, optional (default='JMI')

        A string that specifies the mutual information measure. Possible options are:

        - **JMI** : Joint Mutual Information [3]_
        - **JMIM** : Joint Mutual Information Maximisation [4]_
        - **MRMR** : Max-Relevance Min-Redundancy [5]_

    k : int, optional (default=5)

        An integer to specify number of samples for the kernel density estimation using KNN. A small integer between
        3 and 10 is recommended. Note that this parameter is applicable only if `num_bins` is set to 0.

    n_features : int or string, optional (default='auto')

        A positive integer number that specifies the number of features to be selected.
        If 'auto' is used, then this number will be determined automatically.

    categorical : boolean, optional (default=True)

        If `categorical` is `True`, then the dependent variable is assumed to be a categorical class label. Otherwise, the dependent variable is assumed to be a continuous variable.

    n_jobs : int, optional (default=1)

        The number of threads allowed for the computations. If `n_jobs` :math:`<= 0`, then it will be determined internally.

    verbose : int, optional (default=0)

        An integer number to specify the verbosity level. Available options are:

        - 0: Disable output.
        - 1: Display summary results.
        - 2: Displays all results.

    scale_data : boolean, optional (default=True)

        An boolean flag that specifies if the input data input shall be scaled first.

    num_bins : int, optional (default=0)

        An integer number to specify the number of bins allowed for data binning (bucketing). If :math:`0`, then data binning will be disabled, and KNN will be used for kernal density estimation.


    random_state : int, optional (default=0)

        An integer number that specifies the random seed number.

    max_roundings : int, optional (default=0)

        An positive integer to specify the number of iterations allowed for the rounding solution process in [2]_. If `max_roundings` is 0, then this number will be determined internally.


    Attributes
    ----------

    n_features_ : int

        The number of selected features.

    feature_importances_ : array of shape n_features

        The feature importance scores of the selected features.


    Examples
    --------

    This implementation follows the scikit-learn API convention:

    - ``fit(X, y)``
    - ``transform(X)``
    - ``fit_transform(X, y)``


    References
    ----------

    .. [1] T Naghibi, S Hoffmann and B Pfister, "A semidefinite programming based search strategy for feature selection with mutual information measure", IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(8), pp. 1529--1541, 2015. [`Pre-print <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.739.8516&rep=rep1&type=pdf>`_]
    .. [2] M Goemans and D Williamson, "Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming", J. ACM, 42(6), pp. 1115--1145, 1995 [`Pre-print <http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf>`_]
    .. [3] H Yang and J Moody, "Data Visualization and Feature Selection: New Algorithms for Nongaussian Data", NIPS 1999. [`Pre-print <https://papers.nips.cc/paper/1779-data-visualization-and-feature-selection-new-algorithms-for-nongaussian-data.pdf>`_]
    .. [4] M Bennasar, Y Hicks, abd R Setchi, "Feature selection using Joint Mutual Information Maximisation", Expert Systems with Applications, 42(22), pp. 8520--8532, 2015 [`pre-print <https://core.ac.uk/download/pdf/82448198.pdf>`_]
    .. [5] H Peng, F Long, and C Ding, "Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy", IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), pp. 1226--1238, 2005. [`Pre-print <http://ranger.uta.edu/~chqding/papers/mRMR_PAMI.pdf>`_]
    .. [6] Colin: Conic-form Linear Optimizer (`www.colinopt.org <www.colinopt.org>`_).
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
                 random_state=0,
                 max_roundings=0):
        # Call base constructor.
        super(MutualInformationConicOptimization, self).__init__(
            method, k, n_features, categorical,
            n_jobs, verbose, scale_data, num_bins)
        self.random_state = random_state
        self.max_roundings = max_roundings

        # Attributes.
        self.n_features_ = 0
        self.feature_importances_ = []

    def fit(self, X, y):
        """
        Fits the feature selection with conic optimization approach.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        # Colin
        import colinpy
        #from colinpy import *
        from colinpy import ClnModel, ClnError

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
            num_features_sel = min(1000, max(1, num_features * 0.2))
        else:
            num_features_sel = min(num_features, self.n_features)
        offdiagonal_param = -0.5 / (num_features_sel - 1)

        # Calculate the number of rounding iterations.
        if self.max_roundings <= 0:
            self.max_roundings = num_features_sel

        # Populate info.
        self._print_init_result(num_features_sel, num_features, offdiagonal_param)

        # Calculate the MI matrix.
        logging.info("Started calculating full MI matrix.")
        if self.categorical:
            Nx = []
            classes = np.unique(y)
            sum_y = {}
            for yi in classes:
                sum_y[yi] = np.sum(y == yi)
            for yi in y.flatten():
                Nx.append(sum_y[yi])

        if True:
            # Parallelize the outer loop - faster.
            Qtri = Parallel(n_jobs=self.n_jobs)(delayed(get_mico_vector)(
                self, self.k, list(range(i, num_features, 1)), s, offdiagonal_param, self._are_data_binned(), Nx if self.categorical else None, 1
            ) for i, s in enumerate(S))

            for i, s in enumerate(S):
                Q[s, list(range(i, num_features, 1))] = Qtri[s]
        else:
            # Parallelize the outer loop - slower.
            for i, s in enumerate(S):
                S2 = list(range(i, num_features, 1))
                Q[s, S2] = get_mico_vector(self, self.k, S2, s, offdiagonal_param, self._are_data_binned(), Nx if self.categorical else None, self.n_jobs)

        # Ensure the MI matrix is symmetric.
        for i in range(num_features):
            for j in range(i + 1, num_features):
                Q[j, i] = Q[i, j]

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

        #-------------------------------------------------------------------#
        # Input optimization model.                                         #
        #                                                                   #
        # See                                                               #
        #   A Semidefinite Programming Based Search Strategy for Feature    #
        #   Selection with Mutual Information Measure, Eq (26)              #
        #-------------------------------------------------------------------#
        logging.info("Started creating semidefinite optimization model.")
        def map_ij_to_k(i, j, size):
            return i * size + j

        def map_k_to_ij(k, size):
            return int(k/size), int(k%size)

        # Cone information.
        semidefinite_cone = list(range((num_features + 1) * (num_features + 1)))

        # Constraint 1.
        row1_idx = []
        row1_val = []
        for i in range(1, num_features + 1):
            for j in range(1, num_features + 1):
                row1_idx += [map_ij_to_k(i, j, num_features + 1)]
                row1_val += [1.0]
        row1_rhs = (2.0 * num_features_sel - num_features) ** 2

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

        # Model.
        CLN_INFINITY = ClnModel.get_infinity()
        model = ClnModel()

        try:
            # Step 1. Create a model and change the parameters.
            # Create an empty model.
            model.create_mdl()

            # Set parameters.
            model.set_ipa("Model/Verbose", 3 if self.verbose >= 2 else self.verbose)
            model.set_ipa("Model/Presolver/PresolveLv", 1)
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
            if not model.has_solution():
                logging.warning("Resolving the problem again.")
                model.set_ipa("Ips/Solver/Type", 3)
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
        logging.info("Started generating covariance matrix.")
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

        # Rounding solution.
        best_diff = math.inf
        best_score = -math.inf
        best_soln = []
        pdf = stats.multivariate_normal(mean_vec, cov_mat)
        num_roundings = 0

        if DEBUG:
            print("Primal soln: {}".format(prim_soln))
            dbg_opt_val = 0
            for i in range(0, num_features + 1):
                for j in range(0, num_features + 1):
                    dbg_opt_val += prim_soln[i * (num_features + 1) + j] * Qu[i, j]
            print("Obj: {}", dbg_opt_val)
            for i in range(0, num_features + 1):
                print(Qu[i, 0:(num_features + 1)])

        logging.info("Started rounding process.")
        logging.info(" - Max roundings : {}".format(self.max_roundings))
        logging.info("{0:>5}{1:>5}{2:15}".format("Iter", "Diff", "      ObjValue"))
        seed = self.random_state
        while num_roundings < self.max_roundings:
            sampled_pt = pdf.rvs(1, random_state=seed)
            seed += 1
            ref_pt = sampled_pt[0]
            if abs(ref_pt) < 0.1:
                #print("ref_pt", ref_pt)
                continue

            # Solution rounding.
            if ref_pt >= 0:
                curr_soln = [i for i, e in enumerate(sampled_pt[1:len(sampled_pt)]) if e >= 0]
            else:
                curr_soln = [i for i, e in enumerate(sampled_pt[1:len(sampled_pt)]) if e <= 0]
            curr_soln_binary = self._convert_idx_array_to_mask_array(curr_soln, num_features)

            msg = ""
            if True:
                # Calculate feature difference.
                curr_features_sel = np.sum(curr_soln_binary)
                curr_diff = abs(curr_features_sel - num_features_sel)
                curr_score = self._calc_xQx(Q, curr_soln_binary)
                # Update statistics.
                if curr_diff < best_diff:
                    best_diff = curr_diff
                    best_score = curr_score
                    best_soln = curr_soln
                    msg += "*"
                elif curr_diff == best_diff and best_diff == 0:
                    # Check objective value.
                    # Update statistics.
                    if curr_score > best_score:
                        best_score = curr_score
                        best_soln = curr_soln
                        msg += "**"

            else:
                # Calculate the score.
                curr_score = self._calc_xQx(Q, curr_soln_binary)

                # Update statistics.
                if curr_score > best_score or best_score == -math.inf:
                    best_score = curr_score
                    best_soln = curr_soln
                    msg += "*"

            # Populate information.
            num_roundings += 1
            logging.info("{0:>5}{1:>5}{2:+14.6E} {3}".format(num_roundings, int(curr_diff), curr_score, msg))

        #-------------------------------------------------------------------#
        # Generate feature importance scores.                               #
        #-------------------------------------------------------------------#
        best_soln, best_score, feature_importances, feature_raking = \
            self._generate_feature_importances(best_soln, Q, num_features)

        #-------------------------------------------------------------------#
        # Save results                                                      #
        #-------------------------------------------------------------------#
        self.n_features_ = len(best_soln)
        self.feature_importances_ = feature_importances
        self._support_mask = np.zeros(num_features, dtype=np.bool)
        self._support_mask[best_soln] = True

        self._print_end_result(num_features_sel, num_features, best_soln)

        return self

    def _print_init_result(self, num_features_sel, num_features, offdiagonal_param):
        logging.info("Started MICO.")
        logging.info(" - Method        : {}".format(self.method))
        logging.info(" - Num. threads  : {}".format(self.n_jobs))
        logging.info(" - Tot. features : {}".format(num_features))
        logging.info(" - Sel. features : {}".format(num_features_sel))
        logging.info(" - Off-diag param: {}".format(offdiagonal_param))

    def _print_end_result(self, num_features_sel, num_features, best_soln):
        logging.info("Done MICO.")
        logging.info(" - Total feat.   : {}".format(num_features))
        logging.info(" - Target feat.  : {}".format(num_features_sel))
        logging.info(" - Actual feat.  : {}".format(len(best_soln)))


def test_mi():
    import pandas as pd
    import numpy as np
    from pyitlib import discrete_random_variable as drv

    #X = np.array((1.0, 2.0, 1.0, 2.0, 4.12, 4.123, 6.123))
    #print(drv.entropy(X))
    X = np.array((1.0, 2.0, 1.0, 2.0, 4.123, 4.123, 10.123, 1.1))
    #print(drv.entropy(X))
    #print(pd.qcut(np.array(X), 4, retbins=True))
    #print(pd.qcut(np.array(X), 4, retbins=True)[1])

    if False:
        # Test bin.
        print("=" * 80)
        print("Testing bin")
        print("-" * 80)
        num_bins = 3

        # - Binned with pandas.
        pd_bins = pd.cut(np.array(X), num_bins, retbins=True)[1]
        labels = [i for i in range(num_bins)]
        pd_data = np.array(pd.cut(X, bins=pd_bins , labels=labels, include_lowest=True))
        print("Binned with PD.")
        print(" - Bin  : {}".format(pd_bins))
        print(" - Data : {}".format(pd_data))
        #print(type(pd.cut(X, bins=bins, labels=labels, include_lowest=True)))

        # - Binned with NP.
        np_bin_res = np.histogram(X, bins=num_bins)
        np_data = np_bin_res[0]
        np_bins = np_bin_res[1]
        print("Binned with NP.")
        print(" - Bin  : {}".format(np_bins))
        print(" - Data : {}".format(np_data))


    # - Binned with SKL.
    X = np.array([[ -3., 5., 15 ],
                  [ 0.,  6., 14 ],
                  [ 6.,  3., 11 ],
                  [ 16., -3.,13 ],
                  [ 12., -3.,10 ],
                  [ 16.4, -2.,11.2 ],
                  [ 26., 3., 211]
                 ]
    )
    skl_data = bin_data(X, num_bins=3)
    print("Binned with SKL.")
    print(" - Num features : {}".format(X.shape[1]))
    #print(" - Bin  : {}".format(pd_bins))
    print(" - Data :\n {}".format(skl_data))

    print("Scaled dense matrix with SKL.")
    den_data = scale_data(X)
    print(" - Num features : {}".format(den_data.shape[1]))
    print(" - Data :\n {}".format(den_data))
    print(" - Mean :\n {}".format(den_data.mean(axis=0)))
    print(" - Std  :\n {}".format(den_data.std(axis=0)))

    print("Scaled sparse matrix with SKL.")
    csc_data = sparse.csc_matrix(X)
    csr_data = csc_data.tocsr()
    spa_data = scale_data(csr_data)
    print(" - Num features : {}".format(spa_data.shape[1]))
    print(" - Mean :\n {}".format(spa_data.mean(axis=0)))


    print("=" * 80)

    #h = np.histogram2d(X, Y, bins=bins)[0]
    #print(X)

    #print(np.array(df['qcut4']))
    #print(drv.entropy(data))
    #print(np.hstack(np.array([X, X])))


def scale_data(X):
    """"
    Scale the data using SKL.

    References
    ----------
    [1] http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

    # X must be stored in a row-wise order.
    if sparse.issparse(X):
        scalar = MaxAbsScaler()
    else:
        scalar = StandardScaler()#MinMaxScaler() # StandardScaler

    return scalar.fit_transform(X)


def bin_data(X, num_bins):
    """"
    Bin the data using SKL.
    """
    # X must be stored in a row-wise order.
    from sklearn.preprocessing import KBinsDiscretizer

    if type(num_bins) is int:
        #bin_set = [ min(num_bins, len(array))]
        # @todo Check unique value in each feature.
        bin_set = [ num_bins for i in range(X.shape[1]) ]
    else:
        bin_set = num_bins

    est = KBinsDiscretizer(n_bins=bin_set, encode='ordinal', strategy='uniform').fit(X)

    return est.transform(X)


if __name__ == '__main__':
    test_mi()