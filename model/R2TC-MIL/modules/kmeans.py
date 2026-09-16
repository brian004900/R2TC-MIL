"""CRR k-means (euclidean / cosine) for cluster re-weighting."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch


def initialize(X: torch.Tensor, num_clusters: int) -> torch.Tensor:
    num_samples = len(X)
    k = min(num_clusters, num_samples)
    indices = np.random.choice(num_samples, k, replace=False)
    return X[indices].clone()


def pairwise_distance(data1, data2, device=torch.device("cpu"), tqdm_flag=False):
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    return dis.sum(dim=-1).squeeze(-1) if dis.dim() > 2 else dis.sum(dim=-1)


def pairwise_cosine(data1, data2, device=torch.device("cpu"), tqdm_flag=False):
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    A_normalized = A / (A.norm(dim=-1, keepdim=True) + 1e-8)
    B_normalized = B / (B.norm(dim=-1, keepdim=True) + 1e-8)
    cosine = A_normalized * B_normalized
    return 1 - cosine.sum(dim=-1).squeeze(-1)


def kmeans(
    X,
    num_clusters,
    distance="euclidean",
    cluster_centers=None,
    tol=1e-4,
    tqdm_flag=False,
    iter_limit=50,
    device=torch.device("cpu"),
    **kwargs,
):
    """
    Returns:
        choice_cluster: (N,) long
        initial_state: (K, D)
    """
    del kwargs  # ignore unused extras (gamma_for_soft_dtw, n_init, ...)
    if distance == "euclidean":
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == "cosine":
        pairwise_distance_function = partial(pairwise_cosine, device=device, tqdm_flag=tqdm_flag)
    else:
        raise NotImplementedError(f"unsupported distance={distance}")

    X = X.float().to(device)
    n = X.size(0)
    k = min(int(num_clusters), n)
    if k < 1:
        raise ValueError("empty feature for kmeans")

    if cluster_centers is None or (
        isinstance(cluster_centers, (list, tuple)) and len(cluster_centers) == 0
    ):
        initial_state = initialize(X, k)
    elif torch.is_tensor(cluster_centers) and cluster_centers.numel() > 0 and cluster_centers.any():
        initial_state = cluster_centers.to(device).float()
        if initial_state.size(0) != k:
            initial_state = initialize(X, k)
        else:
            dis = pairwise_distance_function(X, initial_state)
            choice_points = torch.argmin(dis, dim=0)
            initial_state = X[choice_points]
    else:
        initial_state = initialize(X, k)

    initial_state = initial_state.to(device)
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()
        for index in range(k):
            selected = torch.nonzero(choice_cluster == index, as_tuple=False).squeeze(-1)
            if selected.numel() == 0:
                selected = torch.randint(n, (1,), device=device)
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)
        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
        iteration += 1
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break
    return choice_cluster.long(), initial_state


def kmeans_predict(
    X,
    cluster_centers,
    distance="euclidean",
    device=torch.device("cpu"),
    tqdm_flag=False,
    **kwargs,
):
    del kwargs
    if distance == "euclidean":
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == "cosine":
        pairwise_distance_function = partial(pairwise_cosine, device=device, tqdm_flag=tqdm_flag)
    else:
        raise NotImplementedError(f"unsupported distance={distance}")

    X = X.float().to(device)
    cluster_centers = cluster_centers.to(device).float()
    dis = pairwise_distance_function(X, cluster_centers)
    return torch.argmin(dis, dim=1).long()
