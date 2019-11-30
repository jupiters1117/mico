"""
MICO: Mutual Information and Conic Optimization for feature selection.

Github repo : https://github.com/jupiters1117/mico
Author      : KuoLing Huang <jupiters1117@gmail.com>
License     : BSD 3 clause


Note
----
MICO is heavily inspired from MIFS by Daniel Homola:

Github repo : https://github.com/danielhomola/mifs
Author      : Daniel Homola <dani.homola@gmail.com>
License     : BSD 3 clause
"""
from mico import MutualInformationForwardSelection, MutualInformationBackwardSelection, MutualInformationConicOptimization
from sklearn.datasets import make_classification, make_regression
import numpy as np 
import logging
import argparse


def setup_logging(level):
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p', level=level)


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


def test_mifs():
    # variables for dataset
    s = 200  # Num rows
    f = 100 # Num cols
    i = int(.1 * f)  # Proportion of the relevant features
    r = int(.05 * f)  # Proportion of the redundant features
    c = 2

    method = 'JMIM'
    num_bins = 0
    scale_data = False
    k = max(1, int(f * 0.25))
    verbose = 2
    early_stop_steps = 10
    n_jobs = 4

    print("Parameters.")
    print(" - method      : {}".format(method))
    print(" - num_bins    : {}".format(num_bins))
    print(" - scale_data  : {}".format(scale_data))
    print(" - k           : {}".format(k))
    print(" - verbose     : {}".format(verbose))
    print(" - n_jobs      : {}".format(n_jobs))

    # simulate dataset with discrete class labels in y
    X, y = make_classification(n_samples=s, n_features=f, n_informative=i,
                               n_redundant=r, n_clusters_per_class=c,
                               random_state=1, shuffle=False)
    #print(X)
    #print(X.shape)
    #print(sum(y) / len(y))

    # perform feature selection
    mifs = MutualInformationForwardSelection(
        method=method, verbose=verbose, k=k, categorical=True, num_bins=num_bins, scale_data=scale_data,
        n_jobs=n_jobs, early_stop_steps=early_stop_steps)
    mifs.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(mifs.get_support())[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))
    #print("Selected features: {}".format(np.where(mico.get_support())[0]))
    #print(mifs.get_support())
    #print(mifs.feature_importances_)
    #print(mifs.ranking_)

    # simulate dataset with continuous y
    X, y = make_regression(n_samples=s, n_features=f, n_informative=i,
                           random_state=0, shuffle=False)
    # perform feature selection
    mico = MutualInformationForwardSelection(
        method=method, verbose=verbose, k=k, categorical=False, num_bins=num_bins, scale_data=scale_data,
        n_jobs=n_jobs, early_stop_steps=early_stop_steps)
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
    n_jobs = 4

    method = 'JMIM'
    num_bins = 0
    scale_data = False
    k = max(1, int(f * 0.25))
    verbose = 2
    early_stop_steps = 0

    print("Parameters.")
    print(" - method      : {}".format(method))
    print(" - num_bins    : {}".format(num_bins))
    print(" - scale_data  : {}".format(scale_data))
    print(" - k           : {}".format(k))
    print(" - verbose     : {}".format(verbose))
    print(" - n_jobs      : {}".format(n_jobs))

    # simulate dataset with discrete class labels in y
    X, y = make_classification(n_samples=s, n_features=f, n_informative=i,
                               n_redundant=r, n_clusters_per_class=c,
                               random_state=1, shuffle=False)
    #print(X)
    #print(X.shape)
    #print(sum(y) / len(y))

    # perform feature selection
    mibs = MutualInformationBackwardSelection(
        method=method, verbose=verbose, k=k, categorical=True, num_bins=num_bins, scale_data=scale_data,
        n_jobs=n_jobs, early_stop_steps=early_stop_steps)
    mibs.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(mibs.get_support())[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))
    #print("Selected features: {}".format(np.where(mibs.get_support())[0]))
    #print(mibs.get_support())
    #print(mibs.feature_importances_)

    # simulate dataset with continuous y
    X, y = make_regression(n_samples=s, n_features=f, n_informative=i,
                           random_state=0, shuffle=False)
    # perform feature selection
    mico = MutualInformationBackwardSelection(
        method=method, verbose=verbose, k=k, categorical=False, num_bins=num_bins, scale_data=scale_data,
        n_jobs=n_jobs, early_stop_steps=early_stop_steps)
    mico.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(mico.get_support())[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))


def test_mico():
    # variables for dataset
    s = 200  # Num rows
    f = 100 # Num cols
    i = int(.1 * f)  # Proportion of the relevant features
    r = int(.05 * f)  # Proportion of the redundant features
    c = 2 # Classes

    method = 'JMIM'
    num_bins = 0
    scale_data = False
    k = max(1, int(f * 0.25))
    verbose = 2
    n_features = int(f * 0.20)
    #n_features = 8#int(f / 2)
    max_rounds = 0
    n_jobs = 4

    print("Parameters.")
    print(" - method      : {}".format(method))
    print(" - num_bins    : {}".format(num_bins))
    print(" - scale_data  : {}".format(scale_data))
    print(" - k           : {}".format(k))
    print(" - verbose     : {}".format(verbose))
    print(" - n_features  : {}".format(n_features))
    print(" - max_rounds  : {}".format(max_rounds))
    print(" - n_jobs      : {}".format(n_jobs))

    # simulate dataset with discrete class labels in y
    X, y = make_classification(n_samples=s, n_features=f, n_informative=i,
                               n_redundant=r, n_clusters_per_class=c,
                               random_state=1, shuffle=False)
    #print(X)
    #print(X.shape)
    #print(sum(y) / len(y))

    # perform feature selection
    mico = MutualInformationConicOptimization(
        method=method, verbose=verbose, k=k, categorical=True, num_bins=num_bins, scale_data=scale_data,
        n_jobs=n_jobs, n_features=n_features, max_roundings=max_rounds)
    mico.fit(X, y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(mico.get_support())[0], i, r)
    print('Sensitivity: ' + str(sens) + '    Precision: ' + str(prec))
    #print("Selected features: {}".format(np.where(mico.get_support())[0]))
    #print(mico.feature_importances_)

    # simulate dataset with continuous y
    X, y = make_regression(n_samples=s, n_features=f, n_informative=i,
                           random_state=0, shuffle=False)
    # perform feature selection
    mico = MutualInformationConicOptimization(
        method=method, verbose=verbose, k=k, categorical=False, num_bins=num_bins, scale_data=scale_data,
        n_jobs=n_jobs, n_features=n_features, max_roundings=max_rounds)
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
