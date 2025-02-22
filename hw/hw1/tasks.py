import numpy as np


# 2 points
def euclidean_distance(X, Y) -> np.ndarray:
    """
    Compute element wise euclidean distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the Euclidean distance between the corresponding pair of vectors from the arrays X and Y
    """
    X_ = X[:, np.newaxis, :] # X_ shape M, 1, K
    Y_ = Y[np.newaxis, :, :] # Y_ shape 1, N, K
    return np.sqrt(np.square(X_ - Y_).sum(axis=2))


# 2 points
def cosine_distance(X, Y) -> np.ndarray:
    """
    Compute element wise cosine distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the cosine distance between the corresponding pair of vectors from the arrays X and Y
    """
    X_norm = np.linalg.norm(X, axis=1).reshape(-1, 1)
    Y_norm = np.linalg.norm(Y, axis=1).reshape(-1, 1)
    return 1 - X @ Y.T / (X_norm @ Y_norm.T)


# 1 point
def manhattan_distance(X, Y) -> np.ndarray:
    """
    Compute element wise manhattan distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the manhattan distance between the corresponding pair of vectors from the arrays X and Y
    """
    return np.abs((X[:, np.newaxis, :] - Y[np.newaxis, :, :])).sum(axis=2)