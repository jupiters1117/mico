"""
Conic optimization approach for feature selection
=================================================

An introductory example that demonstrates how to perform feature selection using :class:`mico.MutualInformationConicOptimization`.
"""
from mico import MutualInformationConicOptimization
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification, make_regression


def test_mico_classification():

    print("=" * 80)
    print("Start classification example.")
    print("=" * 80)

    X, y = make_classification(n_samples=75, n_features=30, n_classes=2)

    # Perform feature selection.
    mico = MutualInformationConicOptimization(verbose=2, categorical=True, n_features=15)
    mico.fit(X, y)

    print("-" * 80)
    print("Populate results.")
    # Populate selected features.
    print(" - Selected features: \n{}".format(mico.get_support()))
    # Populate feature importance scores.
    print(" - Feature importance scores: \n{}".format(mico.feature_importances_))
    # Call transform() on X to filter it down to selected features.
    X_filtered = mico.transform(X)
    print(" - X_filtered: \n{}".format(X_filtered))


def test_mico_regression():

    print("=" * 80)
    print("Start regression example.")
    print("=" * 80)

    X, y = make_regression(n_samples=75, n_features=30, n_targets=1)

    # Perform feature selection.
    mico = MutualInformationConicOptimization(verbose=2, num_bins=0, categorical=False, n_features=15)
    mico.fit(X, y)

    print("-" * 80)
    print("Populate results.")
    # Populate selected features.
    print(" - Selected features: \n{}".format(mico.get_support()))
    # Populate feature importance scores.
    print(" - Feature importance scores: \n{}".format(mico.feature_importances_))
    # Call transform() on X to filter it down to selected features.
    X_filtered = mico.transform(X)
    print(" - X_filtered: \n{}".format(X_filtered))


if __name__ == '__main__':
    test_mico_classification()
    test_mico_regression()
