"""
Example showing the use of the mico module.
"""
from mico import MutualInformationForwardSelection, MutualInformationBackwardSelection, MutualInformationConicOptimization
from sklearn.datasets import make_classification, make_regression
import numpy as np 
import pyitlib
#from mico import get_entropy
#from ..mico.mico_utils import get_entropy
#import mico.mi
#import sys
import math
#sys.path.append("..")
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import copy
from scipy import sparse

import logging
import argparse

# Python code to generate
# random numbers and
# append them to a list
import random


def setup_logging(level):
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p', level=level)


# Function to generate
# and append them
# start = starting range,
# end = ending range
# num = number of
# elements needs to be appended
def get_rand_list(start, end, num):
    res = []

    for j in range(num):
        res.append(random.randint(start, end))

    return res


def check_selection(selected, i, r):
    """
    Check FN, FP, TP ratios among the selected features.
    """
    # reorder selected features
    try:
        selected = set(selected)
        all_f = set(range(i+r))
        TP = len(selected.intersection(all_f))
        FP = len(selected - all_f)
        FN =  len(all_f - selected)
        if (TP+FN) > 0: 
            sens = TP/float(TP + FN)
        else:
            sens = np.nan
        if (TP+FP) > 0:
            prec =  TP/float(TP + FP)   
        else:
            prec = np.nan
    except:
        sens = np.nan
        prec = np.nan
    return sens, prec
    

def test_mi():
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


def test_mifs():
    # variables for dataset
    s = 2000  # Num rows
    f = 1000 # Num cols
    i = int(.1 * f)  # Proportion of the relevant features
    r = int(.05 * f)  # Proportion of the redundant features
    c = 2

    method = 'JMIM'
    num_bins = 0
    scale_data = True
    k = max(1, int(f * 0.25))
    verbose = 2
    early_stop_steps = 10

    print("Parameters.")
    print(" - method      : {}".format(method))
    print(" - num_bins    : {}".format(num_bins))
    print(" - scale_data  : {}".format(scale_data))
    print(" - k           : {}".format(k))
    print(" - verbose     : {}".format(verbose))

    # simulate dataset with discrete class labels in y
    X, y = make_classification(n_samples=s, n_features=f, n_informative=i,
                               n_redundant=r, n_clusters_per_class=c,
                               random_state=1, shuffle=False)
    print(X)
    print(X.shape)
    print(sum(y) / len(y))

    # perform feature selection
    mico = MutualInformationForwardSelection(
        method=method, verbose=verbose, k=k, categorical=True, num_bins=num_bins, scale_data=scale_data, n_jobs=1, early_stop_steps=early_stop_steps)
    mico.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(mico.get_support())[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))
    #print("Selected features: {}".format(np.where(mico.get_support())[0]))
    # print(mico.get_support())

    # simulate dataset with continuous y
    X, y = make_regression(n_samples=s, n_features=f, n_informative=i,
                           random_state=0, shuffle=False)
    # perform feature selection
    mico = MutualInformationForwardSelection(
        method=method, verbose=verbose, k=k, categorical=False, num_bins=num_bins, scale_data=scale_data, n_jobs=1, early_stop_steps=early_stop_steps)
    mico.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(mico.get_support())[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))


def test_mibs():
    # variables for dataset
    s = 200  # Num rows
    f = 100 # Num cols
    i = int(.1 * f)  # Proportion of the relevant features
    r = int(.05 * f)  # Proportion of the redundant features
    c = 2

    method = 'JMI'
    num_bins = 0
    scale_data = True
    k = max(1, int(f * 0.25))
    verbose = 2
    early_stop_steps = 10

    print("Parameters.")
    print(" - method      : {}".format(method))
    print(" - num_bins    : {}".format(num_bins))
    print(" - scale_data  : {}".format(scale_data))
    print(" - k           : {}".format(k))
    print(" - verbose     : {}".format(verbose))

    # simulate dataset with discrete class labels in y
    X, y = make_classification(n_samples=s, n_features=f, n_informative=i,
                               n_redundant=r, n_clusters_per_class=c,
                               random_state=1, shuffle=False)
    print(X)
    print(X.shape)
    print(sum(y) / len(y))

    # perform feature selection
    mico = MutualInformationBackwardSelection(
        method=method, verbose=verbose, k=k, categorical=True, num_bins=num_bins, scale_data=scale_data, n_jobs=1, early_stop_steps=early_stop_steps)
    mico.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(mico.get_support())[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))
    #print("Selected features: {}".format(np.where(mico.get_support())[0]))

    # simulate dataset with continuous y
    X, y = make_regression(n_samples=s, n_features=f, n_informative=i,
                           random_state=0, shuffle=False)
    # perform feature selection
    mico = MutualInformationBackwardSelection(
        method=method, verbose=verbose, k=k, categorical=False, num_bins=num_bins, scale_data=scale_data, n_jobs=1, early_stop_steps=early_stop_steps)
    mico.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(mico.get_support())[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))


def test_mico():
    # variables for dataset
    s = 500  # Num rows
    f = 200 # Num cols
    i = int(.1 * f)  # Proportion of the relevant features
    r = int(.05 * f)  # Proportion of the redundant features
    c = 2# Classes

    method = 'JMIM'
    num_bins = 0
    scale_data = True
    k = max(1, int(f * 0.25))
    verbose = 2
    n_features = int(f * 0.25)
    #n_features = 8#int(f / 2)

    print("Parameters.")
    print(" - method      : {}".format(method))
    print(" - num_bins    : {}".format(num_bins))
    print(" - scale_data  : {}".format(scale_data))
    print(" - k           : {}".format(k))
    print(" - verbose     : {}".format(verbose))
    print(" - n_features  : {}".format(n_features))

    # simulate dataset with discrete class labels in y
    X, y = make_classification(n_samples=s, n_features=f, n_informative=i,
                               n_redundant=r, n_clusters_per_class=c,
                               random_state=1, shuffle=False)
    print(X)
    print(X.shape)
    print(sum(y) / len(y))

    # perform feature selection
    mico = MutualInformationConicOptimization(
        method=method, verbose=verbose, k=k, categorical=True, num_bins=num_bins, scale_data=scale_data, n_jobs=4, n_features=n_features)
    mico.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(mico.get_support())[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))
    #print("Selected features: {}".format(np.where(mico.get_support())[0]))

    # simulate dataset with continuous y
    X, y = make_regression(n_samples=s, n_features=f, n_informative=i,
                           random_state=0, shuffle=False)
    # perform feature selection
    mico = MutualInformationConicOptimization(
        method=method, verbose=verbose, k=k, categorical=False, num_bins=num_bins, scale_data=scale_data, n_jobs=1, n_features=n_features)
    mico.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(mico.get_support())[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))


def test_colin():
    """
    /**
     *  Description
     *  -----------
     *
     *  Semidefinite optimization (row-wise input).
     *
     *  Model
     *  -----
     *
     *  Minimize
     *  obj: 2 x0 + 1 x1 + 1 x2 + 2 x3
     *       + 3 x4 + 1 x6 + 2 x8 + 1 x10 + 3 x12
     *  Subject To
     *    c1 : 3 x0 + 1 x1 + 1 x2 + 3 x3 + 1 x13 = 1
     *    c2 : 3 x4 + 1 x6 + 4 x8 + 1 x10 + 5 x12
     *       + 1 x14 = 2
     *  Bounds
     *    x0 free
     *    x1 free
     *    x2 free
     *    x3 free
     *    x4 free
     *    x5 free
     *    x6 free
     *    x7 free
     *    x8 free
     *    x9 free
     *    x10 free
     *    x11 free
     *    x12 free
     *    x13 >= 0
     *    x14 >= 0
     *  Semidefinites
     *    s1: [x0, x1,
     *         x2, x3] >= 0
     *    s2: [x4, x5, x6,
     *         x7, x8, x9,
     *         x10, x11, x12] >= 0 (second semidefinite block)
     *  End
     */
    """
    # Colin
    #from colinpy import *
    from colinpy import ClnModel, ClnError

    row1_idx = [ 0,   1,   2,   3,   13  ]
    row1_val = [ 3.0, 1.0, 1.0, 3.0, 1.0 ]
    row2_idx = [ 4,   6,   8,   10,  12,   14 ]
    row2_val = [ 3.0, 1.0, 4.0, 1.0, 5.0, 1.0 ]
    semidefinite_cone1 = [ 0, 1, 2, 3 ]
    semidefinite_cone2 = [ 4, 5, 6, 7, 8, 9, 10, 11, 12 ]

    CLN_INFINITY = ClnModel.get_infinity()
    model = ClnModel()

    try:
        # Step 1. Create a model and change the parameters.
        # Create an empty model.
        model.create_mdl()
        # Set verbose level.
        model.set_ipa("Model/Verbose", 3)

        # Step 2. Input model.
        # Change to minimization problem.
        model.set_min_obj_sense()
        # Add variables.
        # First block.
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 2.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 1.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 1.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 2.0, [], [])
        # Second block.
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 3.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 0.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 1.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 0.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 2.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 0.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 1.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 0.0, [], [])
        model.add_col(-CLN_INFINITY, CLN_INFINITY, 3.0, [], [])
        # Linear variables.
        model.add_col(0.0, CLN_INFINITY, 0.0, [], [])
        model.add_col(0.0, CLN_INFINITY, 0.0, [], [])

        # Add constraints.
        # Note that the nonzero elements are inputted in a row-wise order here.
        model.add_row(1.0, 1.0, row1_idx, row1_val)
        model.add_row(2.0, 2.0, row2_idx, row2_val)

        # Add semidefinite conic constraints.
        model.add_semi_definite_cone(semidefinite_cone1)
        model.add_semi_definite_cone(semidefinite_cone2)

        # Step 3. Solve the problem and populate the result.
        # Solve the problem.
        model.solve_prob()

    except ClnError as e:
        print("Received Colin exception.")
        print(" - Explanation   : {}".format(e.message))
        print(" - Code          : {}".format(e.code))
    except Exception as e:
        print("Received exception.")
        print(" - Explanation   : {}".format(e))
    finally:
        # Step 4. Display the result and free the model.
        model.display_results()
        model.free_mdl()


if __name__ == '__main__':

    # Register arguments.
    parser = argparse.ArgumentParser(description='Process MICO.')
    parser.add_argument('job', type=str)
    args = parser.parse_args()

    # Greeting.
    setup_logging('INFO')
    logging.info("Started MICO.")
    logging.info(" - Job            : {0}".format(args.job))

    try:
        if args.job == "mifs":
            test_mifs()
        elif args.job == "mibs":
            test_mibs()
        elif args.job == "mico":
            test_mico()
        elif args.job == "colin":
            test_colin()
        else:
            logging.info("<ERR>: Unknown command [{0}].".format(args.job))
            logging.info("     : Available options are:")
            logging.info("     :  - [mifs] MI based forward selection")
            logging.info("     :  - [mibs] MI based backward selection")
            logging.info("     :  - [mico] MI based features selection using coinc optimization")

    except Exception as e:
        # Handle general exception.
        logging.info("<ERR>: Captured unknown error exception.")
        logging.info("<ERR>:  - Type    : {0}".format(type(e)))
        logging.info("<ERR>:  - Message : {0}".format(e))
    finally:
        # Done.
        logging.info("Terminated MICO.")
