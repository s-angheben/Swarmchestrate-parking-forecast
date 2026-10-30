"""Clusters of sensors for the cluster models (sensor_cluster, behavior_cluster).

Features of a block of 15-min bins, per sensor: mean events (log1p) and mean occupancy at each hour of the day
(clock hour, local time): 24 + 24 values; with the type option, + the sensor's type one-hot (T values). k-means on them,
each block scaled so that it weighs the same: events, occupancy (and type). A cluster's series = the mean of its sensors'
series.
"""
import numpy as np
from sklearn.cluster import KMeans


def day_features(E, O, hours):
    """E, O (bins, S), hours (bins,) clock hour of each bin -> (S, 48). An hour without bins (DST day) -> 0."""
    H = np.eye(24)[hours].T                                             # (24, bins)
    n = H.sum(1, keepdims=True).clip(1)
    return np.hstack([np.log1p(H @ E / n).T, (H @ O / n).T])


def fit(F, k):
    """k-means on features F (points, 48 [+ T type one-hot]) -> centroids, numbered by activity (0 = most events),
    and the scale of each column."""
    T = F.shape[1] - 48
    scale = np.repeat([F[:, :24].std(), F[:, 24:48].std()], 24)
    if T:                                                               # type block: total weight = one block of 24
        scale = np.append(scale, np.repeat(F[:, 48:].std() * np.sqrt(T / 24), T))
    c = KMeans(n_clusters=k, n_init=10, random_state=0).fit(F / scale).cluster_centers_
    return c[np.argsort(-c[:, :24].mean(1))], scale


def assign(F, centroids, scale):
    """Nearest centroid of each row of F (S, 48 [+ T]) -> (S,) cluster."""
    return (((F / scale)[:, None] - centroids[None]) ** 2).sum(-1).argmin(1)


def pool(X, labels, k):
    """X (bins, S) -> (bins, k) mean of each cluster's sensors (0 if empty), and the sensors per cluster (k,)."""
    n = np.bincount(labels, minlength=k)
    return X @ np.eye(k)[labels] / n.clip(1), n
