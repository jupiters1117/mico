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
from mico import MutualInformationForwardSelection
import pandas as pd
from sklearn.datasets import load_breast_cancer


def test_mifs():

    # Prepare data.
    data = load_breast_cancer()
    y = data.target
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Perform feature selection.
    mico = MutualInformationForwardSelection(verbose=2, categorical=True, n_jobs=1, n_features=10)
    mico.fit(X, y)

    print("=" * 80)
    print("Populate results.")
    # Populate selected features.
    print(" - Selected features: \n{}".format(mico.get_support()))
    # Populate feature importance scores.
    print(" - Feature importance scores: \n{}".format(mico.feature_importances_))
    # Call transform() on X.
    X_transformed = mico.transform(X)
    print(" - X_transformed: \n{}".format(X_transformed))



if __name__ == '__main__':
    test_mifs()
