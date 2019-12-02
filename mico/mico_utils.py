'''
Non-parametric computation of get_entropy and mutual-information
Adapted by G Varoquaux for code created by R Brette, itself
from several papers (see in the code).
These computations rely on nearest-neighbor statistics
'''
import numpy as np
from pyitlib import discrete_random_variable as drv
from scipy.special import gamma,psi
from scipy import ndimage
from scipy.linalg import det
from sklearn.datasets import make_classification, make_regression
from sklearn.neighbors import NearestNeighbors
import bottleneck as bn


__all__=['get_entropy', 'get_mutual_information', 'get_entropy_gaussian']

EPS = np.finfo(float).eps
DEBUG = False


import random
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


def make_mat_sym(mat):
    return (mat.transpose() + mat) / 2.0


def get_nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    #knn = NearestNeighbors(n_neighbors=k + 1)
    knn = NearestNeighbors(n_neighbors=k, metric='chebyshev')
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor


def get_entropy_gaussian(C):
    '''
    get_entropy of a gaussian variable with covariance matrix C
    '''
    if np.isscalar(C): # C is the variance
        return .5*(1 + np.log(2 * np.pi)) + .5*np.log(C)
    else:
        n = C.shape[0] # dimension
        return .5*n*(1 + np.log(2 * np.pi)) + .5*np.log(abs(det(C)))


def get_entropy_c(X, k=1):
    '''
    Returns the get_entropy of the X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data the get_entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation

    Notes
    -----
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of get_entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = get_nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (np.pi ** (.5*d)) / gamma(.5*d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return d * np.mean(np.log(r + np.finfo(X.dtype).eps)) + np.log(volume_unit_ball) + np.log(n) - np.log(k)
    #return d*np.mean(np.log(r + np.finfo(X.dtype).eps)) + np.log(volume_unit_ball) + psi(n) - psi(k)


def get_entropy_d(x):
    '''
    Returns the get_entropy of the X.

    Parameters
    ----------
    X : array-like, shape (n_samples)
        The data the get_entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation

    Notes
    -----
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of get_entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''
    return drv.entropy(x)


def get_mutual_information_cc(x, y, k=5):
    '''
    Returns the mutual information between two continuous vectors x and y.
    Each variable is a matrix = array(n_samples, n_features)
    where
      n_samples = number of samples
      n_features = number of features

    Parameters
    ----------
    X : array-like, shape (m samples, n features)
        The data used for mutual information calculation.
    Y : array-like, shape (m samples, o features)
        The data used for mutual information calculation.
    k : int, optional (default is 5)
        number of nearest neighbors for density estimation

    Returns
    -------
        Mutual information between two variables.

    Example
    -------
        get_mutual_information((X, Y))
    '''
    all_vars = np.hstack([x, y]) # Stacked all features.
    # I(A, B) = -\sum_{a, b} P(a, b) / (P(a)P(b)) = H(A) + H(B) − H(A,B), where
    # H(A) = -\sum_{a}P(a)log_2 P(a).
    #return sum([get_entropy(X, k=k) for X in variables]) - get_entropy(all_vars, k=k)
    res = get_entropy_c(x, k=k) + get_entropy_c(y, k=k) - get_entropy_c(all_vars, k=k)
    return res


def get_mutual_information_cd(x, y, k, Nx=None):
    """
    Calculates the mutual information between a continuous vector x and a
    discrete class vector y.

    This implementation can calculate the MI between the joint distribution of
    one or more continuous variables (X[:, 1:3]) with a discrete variable (y).

    Thanks to Adam Pocock, the author of the FEAST package for the idea.

    Brian C. Ross, 2014, PLOS ONE
    Mutual Information between Discrete and Continuous Data Sets
    """
    #print("BGN KNN")

    n = x.shape[0]
    classes = np.unique(y)
    knn = NearestNeighbors(n_neighbors=k)
    # distance to kth in-class neighbors
    dist_to_k_neighbors = np.empty(n)

    # number of points within each point's class
    if Nx is None:
        Nx = []
        if False:
            for yi in y:
                #print(np.sum(y == yi))
                Nx.append(np.sum(y == yi))
        else:
            sum_y = {}
            for yi in classes:
                sum_y[yi] = np.sum(y == yi)
            for yi in y.flatten():
                Nx.append(sum_y[yi])

    #print(">> KNN")
    # find the distance of the kth in-class point
    # See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors
    for c in classes:
        mask = np.where(y == c)[0]
        #print(x[mask, :].shape[0])
        if x[mask, :].shape[0] <= k:
            continue
            #raise ValueError("Parameter `k` is too large: current value = {0}, max possible value = {1}.".format(k, x[mask, :].shape[0]-1))
        knn.fit(x[mask, :])
        # Return the distance to the k nearest neighbors from each x of interest.
        dist = knn.kneighbors()[0]
        #dist_to_k_neighbors[mask] = bn.nanmax(dist, axis=1) # Max distance to the k neighbors.
        dist_to_k_neighbors[mask] = dist[:, -1] # The returned distances are already sorted.
        if DEBUG:
            assert((bn.nanmax(dist, axis=1) == dist[:, -1]).all())
        #print(dist)
        #print(bn.nanmax(dist, axis=1), dist[:, -1])
    #print("dist_to_k_neighbors=", dist_to_k_neighbors)

    #print(">> FIT-KNN")
    # find the number of points within the distance of the kth in-class point
    knn.fit(x)
    # Note: this is not supported in SKL, but it will still generate the correct result (turn DEBUG to test the implementation).
    m = knn.radius_neighbors(radius=dist_to_k_neighbors, return_distance=False)
    #print(m)
    #print("DONE KNN")

    if DEBUG:
        print("Warning: Debug mode.")
        m_debug = []
        for i in range(n):
            neighbors = knn.radius_neighbors(X = x[i, :].reshape(1, x.shape[1]), radius=dist_to_k_neighbors[i], return_distance=False)[0]
            #print("i = {0}, n = {1}".format(i, neighbors))
            neighbors = np.delete(neighbors, np.where(neighbors == i))
            m_debug.append(neighbors)
        for i in range(n):
            if sorted(m_debug[i]) != sorted(m[i]):
                raise ValueError("sorted(m_debug[i]) = {0}, sorted(m[i] = {1}".format(sorted(m_debug[i]), sorted(m[i])))
            #print("m = ", sorted(m_debug[i]))
            #print("m2 = ", sorted(m[i]))

    # calculate MI based on Equation 2 in Ross 2014
    m_size = [i.shape[0] for i in m]
    #print("psi(n) = ", psi(n))
    #print("psi(k) = ", psi(k))
    #print("np.mean(psi(Nx)) = ", np.mean(psi(Nx)))
    #print("np.mean(psi(m)) = ", np.mean(psi(m_size)))
    MI = psi(n) - np.mean(psi(Nx)) + psi(k) - np.mean(psi(m_size))
    #print("MI = ", MI)
    #raise ValueError("STOP")
    #print(">> MI")

    return MI


def encode_discrete_x(x):
    if x.shape[1] > 1:
        from sklearn import preprocessing

        # Form a new discrete feature by encoding multiple discrete features. For example,
        # [ 0 1 ]    [ 0_1 ]
        # [ 1 3 ] => [ 1_3 ]
        # [ 0 1 ]    [ 0_1 ]
        x_new_data = [list(map(str, map(lambda r: round(r, 4), x[i, :]))) for i in range(x.shape[0])]
        # print("x_new_data1", x_new_data)
        # print("test=", ','.join(x_new_data[0]))
        x_new_data = list(map(lambda x: ','.join(x), x_new_data))
        # print("x_new_data2", x_new_data)
        # print("ratio=", len(set(x_new_data))/len(x_new_data))

        le = preprocessing.LabelEncoder()
        le.fit(x_new_data)
        x_encoded = le.transform(x_new_data)
        # print("x_encoded=", x_encoded)
        # print("y=", y)

        return x_encoded
    else:
        # No need to encode the feature.
        return x


def get_mutual_information_dd(x, y):
    '''
    Returns the mutual information between two continuous vectors x and y.
    Each variable is a matrix = array(n_samples, n_features)
    where
      n_samples = number of samples
      n_features = number of features

    Parameters
    ----------
    X : array-like, shape (m samples, n features)
        The data used for mutual information calculation.
    Y : array-like, shape (m samples, 1 features)
        The data used for mutual information calculation.
    k : int, optional (default is 5)
        number of nearest neighbors for density estimation

    Returns
    -------
        Mutual information between two variables.

    Example
    -------
        get_mutual_information((X, Y))
    '''
    return drv.information_mutual(x.reshape(len(x)), y.reshape(len(y)), cartesian_product=True)

    # Encoding first.
    #print(x)
    #if x.shape[1] > 1:
    #    x_encoded = encode_discrete_x(x)
    #    return drv.information_mutual(x_encoded, y.reshape(len(y)), cartesian_product=True)
    #else:
    #    # No need to encode the feature.


def get_interaction_information_3way(x1, x2, y, var_type, k=5):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: get_mutual_information((X, Y)), get_mutual_information((X, Y, Z), k=5)
    Note
    ====
    Symmetric, but may be negative/

    '''

    # I(x1; x2; y) = I(x1, x2; y) - I(x1, y) - I(x2; y)
    #              = H(x1, x2) + H(x2, y) + H(x1, y) - H(x1) - H(x2) - H(y) - H(x1, x2, y).
    if var_type == "ccd":
        # Discrete y. All continuous X.
        I_x1x2_y = max(0.0, get_mutual_information_cd(np.hstack([x1, x2]), y, k))
        #print("I_x1x2_y", I_x1x2_y)
        #print("np.hstack([x1, x2])", np.hstack([x1, x2]))
        I_x1_y = max(0.0, get_mutual_information_cd(x1, y, k))
        I_x2_y = max(0.0, get_mutual_information_cd(x2, y, k))
        #print(I_x1x2_y - I_x1_y - I_x2_y)
        return I_x1x2_y - I_x1_y - I_x2_y
    elif var_type == "ccc":
        # Continuous y. All continuous X.
        I_x1x2_y = max(0.0, get_mutual_information_cc(np.hstack([x1, x2]), y, k))
        I_x1_y = max(0.0, get_mutual_information_cc(x1, y, k))
        I_x2_y = max(0.0, get_mutual_information_cc(x2, y, k))
        return I_x1x2_y - I_x1_y - I_x2_y
    if var_type == "ddd":
        # Discrete y. All discrete X.
        x_encoded = encode_discrete_x(np.hstack([x1, x2]))
        I_x1x2_y = max(0.0, get_mutual_information_dd(x_encoded, y))
        I_x1_y = max(0.0, get_mutual_information_dd(x1, y))
        I_x2_y = max(0.0, get_mutual_information_dd(x2, y))
        return I_x1x2_y - I_x1_y - I_x2_y
    elif var_type == "ddc":
        # Continuous y. All discrete X.
        x_encoded = encode_discrete_x(np.hstack([x1, x2]))
        I_x1x2_y = max(0.0, get_mutual_information_cd(y, x_encoded, k))
        I_x1_y = max(0.0, get_mutual_information_cd(y, x1, k))
        I_x2_y = max(0.0, get_mutual_information_cd(y, x2, k))
        return I_x1x2_y - I_x1_y - I_x2_y
    else:
        raise ValueError("Unknown supported var_type {0} in get_interaction_information_3way".format(var_type))


def get_mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.

    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram

    Returns
    -------
    float
        the computed similarity measure
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized get_entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi



###############################################################################
# Tests

def test_entropy():
    # Testing against correlated Gaussian variables
    # (analytical results are known)
    # get_entropy of a 3-dimensional gaussian variable
    rng = np.random.RandomState(0)
    n = 50000
    d = 3
    P = np.array([[1, 0, 0], [0, 1, .5], [0, 0, 1]])
    C = np.dot(P, P.T)
    Y = rng.randn(d, n)
    X = np.dot(P, Y)
    H_th = get_entropy_gaussian(C)
    H_est = get_entropy_c(X.T, k=5)
    # Our estimated get_entropy should always be less that the actual one
    # (get_entropy estimation undershoots) but not too much
    np.testing.assert_array_less(H_est, H_th)
    print(H_th*0.75, H_est)
    np.testing.assert_array_less(.75*H_th, H_est)


def test_mutual_information():
    # Mutual information between two correlated gaussian variables
    # get_entropy of a 2-dimensional gaussian variable
    n = 50000
    rng = np.random.RandomState(0)
    #P = np.random.randn(2, 2)
    P = np.array([[1, 0], [0.5, 1]])
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(int(len(X)/2), 2)
    Y = Z[:, 1]
    Y = Y.reshape(int(len(Y)/2), 2)
    # in bits
    print(X.shape)
    print(Y.shape)
    MI_est = get_mutual_information_cc(X, Y, k=5)
    all_vars = np.hstack((X,Y))
    print(all_vars.shape)

    MI_th = (get_entropy_gaussian(C[0, 0])
             + get_entropy_gaussian(C[1, 1])
             - get_entropy_gaussian(C)
            )



    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    MI_est = get_mutual_information_cc(X, Y, k=5)
    MI_th = (get_entropy_gaussian(C[0, 0])
             + get_entropy_gaussian(C[1, 1])
             - get_entropy_gaussian(C)
            )

    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    print((MI_est, MI_th))
    np.testing.assert_array_less(MI_est, MI_th)
    np.testing.assert_array_less(MI_th, MI_est  + .3)


def test_degenerate():
    # Test that our estimators are well-behaved with regards to
    # degenerate solutions
    rng = np.random.RandomState(0)
    x = rng.randn(50000)
    X = np.c_[x, x]
    assert np.isfinite(get_entropy_c(X))
    assert np.isfinite(get_mutual_information_cc(x[:, np.newaxis], x[:,  np.newaxis]))
    assert 2.9 < get_mutual_information_2d(x, x) < 3.1


def test_mutual_information_2d():
    # Mutual information between two correlated gaussian variables
    # get_entropy of a 2-dimensional gaussian variable
    n = 50000
    rng = np.random.RandomState(0)
    #P = np.random.randn(2, 2)
    P = np.array([[1, 0], [.9, .1]])
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    MI_est = get_mutual_information_2d(X.ravel(), Y.ravel())
    MI_th = (get_entropy_gaussian(C[0, 0])
             + get_entropy_gaussian(C[1, 1])
             - get_entropy_gaussian(C)
            )
    print((MI_est, MI_th))
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    np.testing.assert_array_less(MI_est, MI_th)
    np.testing.assert_array_less(MI_th, MI_est  + .2)


def test_mutual_information_misc():
    # variables for dataset
    s = 200  # Num rows
    f = 100  # Num cols
    i = int(.1 * f)  # Proportion of the relevant features
    r = int(.05 * f)  # Proportion of the redundant features
    c = 2

    # simulate dataset with discrete class labels in y
    X, y = make_classification(n_samples=s, n_features=f, n_informative=i,
                               n_redundant=r, n_clusters_per_class=c,
                               random_state=1, shuffle=False)
    print(X.shape)
    #mi._mi_dc(x, y, k)
    get_nearest_distances(X)
    k = int(X.shape[0] * 0.1)
    r = get_nearest_distances(X, k)
    n, d = X.shape
    volume_unit_ball = (np.pi ** (.5 * d)) / gamma(.5 * d + 1)
    print(psi(10))
    print(psi(3))
    #print(np.finfo(X.dtype).eps)


    print("My entropy = ", d * np.mean(np.log(r + np.finfo(X.dtype).eps)) +
            np.log(volume_unit_ball) + np.log(n) - np.log(k))

    print("Current entropy = ", get_entropy_c(X, k=k))


if __name__ == '__main__':
    # Run our tests
    test_entropy()
    test_mutual_information()
    test_degenerate()
    test_mutual_information_2d()
    test_mutual_information_misc()