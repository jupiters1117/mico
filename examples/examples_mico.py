"""
Conic optimization approach for feature selection
=================================================

An introductory example that demonstrates how to perform feature selection using :class:`mico.MutualInformationConicOptimization`.
"""
from mico import MutualInformationConicOptimization
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes


def test_mico_classification():

    print("=" * 80)
    print("Start classification example.")
    print("=" * 80)

    # Prepare data.
    data = load_breast_cancer()
    y = data.target
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Perform feature selection.
    mico = MutualInformationConicOptimization(verbose=2, categorical=True, n_features=7)
    mico.fit(X, y)

    print("-" * 80)
    print("Populate results.")
    # Populate selected features.
    print(" - Selected features: \n{}".format(mico.get_support()))
    # Populate feature importance scores.
    print(" - Feature importance scores: \n{}".format(mico.feature_importances_))
    # Call transform() on X.
    X_transformed = mico.transform(X)
    print(" - X_transformed: \n{}".format(X_transformed))


def test_mico_regression():

    print("=" * 80)
    print("Start regression example.")
    print("=" * 80)

    # Prepare data.
    data = load_diabetes()
    y = data.target
    X = pd.DataFrame(data.data, columns=data.feature_names)
    print(X)
    print(y)

    # Perform feature selection.
    mico = MutualInformationConicOptimization(verbose=2, num_bins=0, categorical=False, n_features=5)
    mico.fit(X, y)

    print("-" * 80)
    print("Populate results.")
    # Populate selected features.
    print(" - Selected features: \n{}".format(mico.get_support()))
    # Populate feature importance scores.
    print(" - Feature importance scores: \n{}".format(mico.feature_importances_))
    # Call transform() on X.
    X_transformed = mico.transform(X)
    print(" - X_transformed: \n{}".format(X_transformed))


if __name__ == '__main__':
    test_mico_classification()
    test_mico_regression()
