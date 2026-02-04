"""
Utilities for generating toy 2D classification datasets with controllable
aleatoric and epistemic uncertainty using Gaussian blob class-conditional data.

Main entry point:
    simulate_dataset(cfg) -> (X_train, y_train, X_test, y_test, meta)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


@dataclass
class OODRegion:
    kind: str  # "ring", "box", "uniform", "different_clusters", "line", "spiral", "grid", "gaussian_noise"
    params: Dict[str, float]
    n: int


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _compute_scores(X: np.ndarray, centers: np.ndarray, tau: float, biases: np.ndarray) -> np.ndarray:
    # X: (N, 2), centers: (K, 2), biases: (K,)
    diffs = X[:, None, :] - centers[None, :, :]
    d2 = np.sum(diffs ** 2, axis=2)
    return -d2 / tau + biases[None, :]


def _sample_labels_from_probs(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Numpy doesn't support vectorized choice with per-row probabilities reliably across versions,
    # so we sample with cumulative probabilities.
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(size=(probs.shape[0], 1))
    return (u < cdf).argmax(axis=1)


def _apply_label_noise(y: np.ndarray, eta: float, num_classes: int, rng: np.random.Generator) -> np.ndarray:
    if eta <= 0:
        return y
    y_noisy = y.copy()
    flip_mask = rng.random(size=len(y)) < eta
    if np.any(flip_mask):
        for idx in np.where(flip_mask)[0]:
            other_classes = [c for c in range(num_classes) if c != y_noisy[idx]]
            y_noisy[idx] = rng.choice(other_classes)
    return y_noisy


def _apply_input_noise(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return X
    return X + rng.normal(loc=0.0, scale=sigma, size=X.shape)


def _normalize_class_priors(priors: Optional[np.ndarray], num_classes: int) -> np.ndarray:
    if priors is None:
        return np.ones(num_classes, dtype=np.float32) / float(num_classes)
    priors = np.asarray(priors, dtype=np.float32)
    if priors.shape[0] != num_classes:
        raise ValueError(f"class_priors must have length {num_classes}, got {priors.shape[0]}")
    if np.any(priors < 0):
        raise ValueError("class_priors must be non-negative.")
    total = float(priors.sum())
    if total <= 0:
        raise ValueError("class_priors must sum to a positive value.")
    return priors / total


def _sample_gaussian_blobs(
    n: int,
    centers: np.ndarray,
    sigma: float,
    class_priors: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    y = rng.choice(len(centers), size=n, replace=True, p=class_priors)
    X = centers[y] + rng.normal(loc=0.0, scale=sigma, size=(n, centers.shape[1]))
    return X.astype(np.float32), y.astype(np.int64)


def _sample_boundary_points(
    n: int,
    centers: np.ndarray,
    blob_sigma: float,
    boundary_spread: float,
    boundary_width: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points along the midlines between class centers.
    
    Args:
        n: Number of boundary points to sample
        centers: Class center coordinates (K, 2)
        blob_sigma: Base blob standard deviation
        boundary_spread: How far points spread along boundary (multiplier of blob_sigma)
        boundary_width: How far points spread perpendicular to boundary (multiplier)
        rng: Random number generator
    
    Returns:
        X: Boundary points (n, 2)
        y: Assigned labels (nearest class)
    """
    n_classes = len(centers)
    pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
    n_per_pair = n // len(pairs)
    remainder = n % len(pairs)
    
    points = []
    for idx, (i, j) in enumerate(pairs):
        n_this_pair = n_per_pair + (1 if idx < remainder else 0)
        
        midpoint = (centers[i] + centers[j]) / 2
        direction = centers[j] - centers[i]
        length = np.linalg.norm(direction)
        direction = direction / length
        perp = np.array([-direction[1], direction[0]])
        
        # Sample along boundary (perpendicular direction) with small noise toward classes
        t = rng.normal(0, blob_sigma * boundary_spread, size=n_this_pair)
        noise = rng.normal(0, blob_sigma * boundary_width, size=n_this_pair)
        
        batch = midpoint + t[:, None] * perp + noise[:, None] * direction
        points.append(batch)
    
    X = np.vstack(points).astype(np.float32)
    
    # Assign labels based on nearest center
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    y = np.argmin(dists, axis=1).astype(np.int64)
    
    return X, y


def _boundary_band_mask(scores: np.ndarray, d0: float) -> np.ndarray:
    # Margin between top two scores; small margin indicates proximity to boundary.
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    return margin <= d0


def _region_mask_box(X: np.ndarray, region: Dict[str, float]) -> np.ndarray:
    return (
        (X[:, 0] >= region["x_min"])
        & (X[:, 0] <= region["x_max"])
        & (X[:, 1] >= region["y_min"])
        & (X[:, 1] <= region["y_max"])
    )


def _region_mask_circle(X: np.ndarray, region: Dict[str, float]) -> np.ndarray:
    cx, cy = region["x_center"], region["y_center"]
    r = region["radius"]
    return (X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2 <= r ** 2


def _apply_undersampling(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    undersampling: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not undersampling:
        return np.ones(len(X), dtype=bool), {"removed_total": 0}

    keep_mask = np.ones(len(X), dtype=bool)
    meta: Dict[str, Any] = {"removed_total": 0, "regions": []}

    # Boundary band undersampling
    boundary_cfg = undersampling.get("boundary_band")
    if boundary_cfg and boundary_cfg.get("enabled", False):
        d0 = float(boundary_cfg.get("d0", 0.1))
        rho = float(boundary_cfg.get("rho", 0.0))
        band_mask = _boundary_band_mask(scores, d0)
        band_indices = np.where(band_mask)[0]
        n_remove = int(np.floor(rho * len(band_indices)))
        if n_remove > 0:
            remove_idx = rng.choice(band_indices, size=n_remove, replace=False)
            keep_mask[remove_idx] = False
        meta["regions"].append(
            {"type": "boundary_band", "d0": d0, "rho": rho, "candidates": len(band_indices), "removed": int(n_remove)}
        )

    # Hole/box undersampling
    for region in undersampling.get("holes", []):
        region_type = region.get("type", "box")
        rho = float(region.get("rho", undersampling.get("rho", 0.0)))
        if region_type == "circle":
            region_mask = _region_mask_circle(X, region)
        else:
            region_mask = _region_mask_box(X, region)

        region_indices = np.where(region_mask & keep_mask)[0]
        n_remove = int(np.floor(rho * len(region_indices)))
        if n_remove > 0:
            remove_idx = rng.choice(region_indices, size=n_remove, replace=False)
            keep_mask[remove_idx] = False
        meta["regions"].append(
            {"type": region_type, "rho": rho, "candidates": len(region_indices), "removed": int(n_remove)}
        )

    removed_total = int((~keep_mask).sum())
    meta["removed_total"] = removed_total
    return keep_mask, meta


def _parse_ood_regions(cfg: Dict[str, Any]) -> List[OODRegion]:
    ood_specs = cfg.get("ood_specs", {}) or {}
    regions: List[OODRegion] = []
    n_ood_total = int(cfg.get("N_ood_test", 0))

    if "ring" in ood_specs:
        ring = ood_specs["ring"]
        n = int(ring.get("n", 0))
        regions.append(OODRegion(kind="ring", params=ring, n=n))

    for box in ood_specs.get("squares", []):
        n = int(box.get("n", 0))
        regions.append(OODRegion(kind="box", params=box, n=n))

    # New OOD types for different generation processes
    if "uniform" in ood_specs:
        uniform = ood_specs["uniform"]
        n = int(uniform.get("n", 0))
        regions.append(OODRegion(kind="uniform", params=uniform, n=n))

    if "different_clusters" in ood_specs:
        clusters = ood_specs["different_clusters"]
        n = int(clusters.get("n", 0))
        regions.append(OODRegion(kind="different_clusters", params=clusters, n=n))

    if "line" in ood_specs:
        line = ood_specs["line"]
        n = int(line.get("n", 0))
        regions.append(OODRegion(kind="line", params=line, n=n))

    if "spiral" in ood_specs:
        spiral = ood_specs["spiral"]
        n = int(spiral.get("n", 0))
        regions.append(OODRegion(kind="spiral", params=spiral, n=n))

    if "grid" in ood_specs:
        grid = ood_specs["grid"]
        n = int(grid.get("n", 0))
        regions.append(OODRegion(kind="grid", params=grid, n=n))

    if "gaussian_noise" in ood_specs:
        noise = ood_specs["gaussian_noise"]
        n = int(noise.get("n", 0))
        regions.append(OODRegion(kind="gaussian_noise", params=noise, n=n))

    if "confident_ood" in ood_specs:
        confident = ood_specs["confident_ood"]
        n = int(confident.get("n", 0))
        regions.append(OODRegion(kind="confident_ood", params=confident, n=n))

    # Allocate if needed
    if n_ood_total > 0 and any(r.n == 0 for r in regions):
        empty_regions = [r for r in regions if r.n == 0]
        base = n_ood_total // max(1, len(empty_regions))
        remainder = n_ood_total - base * len(empty_regions)
        for idx, region in enumerate(empty_regions):
            region.n = base + (1 if idx < remainder else 0)

    return regions


def _sample_ood_points(regions: List[OODRegion], rng: np.random.Generator) -> np.ndarray:
    samples: List[np.ndarray] = []
    for region in regions:
        n = region.n
        if n <= 0:
            continue
        
        if region.kind == "ring":
            r_min = float(region.params["r_min"])
            r_max = float(region.params["r_max"])
            theta = rng.uniform(0.0, 2 * np.pi, size=n)
            r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2, size=n))
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            samples.append(np.stack([x, y], axis=1))
            
        elif region.kind == "box":
            x = rng.uniform(float(region.params["x_min"]), float(region.params["x_max"]), size=n)
            y = rng.uniform(float(region.params["y_min"]), float(region.params["y_max"]), size=n)
            samples.append(np.stack([x, y], axis=1))
            
        elif region.kind == "uniform":
            x_range = region.params.get("x_range", [-2.0, 2.0])
            y_range = region.params.get("y_range", [-2.0, 2.0])
            x = rng.uniform(x_range[0], x_range[1], size=n)
            y = rng.uniform(y_range[0], y_range[1], size=n)
            samples.append(np.stack([x, y], axis=1))
            
        elif region.kind == "different_clusters":
            n_clusters = int(region.params.get("n_clusters", 2))
            cluster_sigma = float(region.params.get("sigma", 0.3))
            cluster_centers = rng.uniform(-1.5, 1.5, size=(n_clusters, 2))
            
            # Sample points from these clusters
            cluster_assignments = rng.integers(0, n_clusters, size=n)
            points = []
            for cluster_idx in cluster_assignments:
                center = cluster_centers[cluster_idx]
                point = rng.normal(center, cluster_sigma, size=(1, 2))
                points.append(point)
            samples.append(np.vstack(points))
            
        elif region.kind == "line":
            x_start = float(region.params.get("x_start", -2.0))
            x_end = float(region.params.get("x_end", 2.0))
            slope = float(region.params.get("slope", 1.0))
            intercept = float(region.params.get("intercept", 0.0))
            noise = float(region.params.get("noise", 0.1))
            
            x = rng.uniform(x_start, x_end, size=n)
            y = slope * x + intercept + rng.normal(0, noise, size=n)
            samples.append(np.stack([x, y], axis=1))
            
        elif region.kind == "spiral":
            n_turns = float(region.params.get("n_turns", 2.0))
            r_max = float(region.params.get("r_max", 1.5))
            noise = float(region.params.get("noise", 0.05))
            
            t = rng.uniform(0, n_turns * 2 * np.pi, size=n)
            r = r_max * t / (n_turns * 2 * np.pi)
            x = r * np.cos(t) + rng.normal(0, noise, size=n)
            y = r * np.sin(t) + rng.normal(0, noise, size=n)
            samples.append(np.stack([x, y], axis=1))
            
        elif region.kind == "grid":
            x_range = region.params.get("x_range", [-2.0, 2.0])
            y_range = region.params.get("y_range", [-2.0, 2.0])
            
            n_grid = int(np.sqrt(n)) + 1
            x_vals = np.linspace(x_range[0], x_range[1], n_grid)
            y_vals = np.linspace(y_range[0], y_range[1], n_grid)
            xx, yy = np.meshgrid(x_vals, y_vals)
            grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
            
            # Randomly sample n points from grid
            indices = rng.choice(len(grid_points), size=min(n, len(grid_points)), replace=False)
            samples.append(grid_points[indices])
            
        elif region.kind == "gaussian_noise":
            mean = region.params.get("mean", [0.0, 0.0])
            std = float(region.params.get("std", 1.0))
            x = rng.normal(mean[0], std, size=n)
            y = rng.normal(mean[1], std, size=n)
            samples.append(np.stack([x, y], axis=1))
            
        elif region.kind == "confident_ood":
            # OOD points that extend far from training but in direction of a class
            training_centers = region.params.get("training_centers")
            if training_centers is None:
                raise ValueError("confident_ood requires training_centers parameter")
            training_centers = np.array(training_centers)
            extension_factor = float(region.params.get("extension_factor", 2.0))
            ood_sigma = float(region.params.get("sigma", 0.2))
            
            points = []
            for _ in range(n):
                # Pick a random training center and extend far in that direction
                center_idx = rng.integers(0, len(training_centers))
                center = training_centers[center_idx]
                # Random direction from center
                angle = rng.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
                # Extend far from center
                base_point = center + direction * extension_factor
                # Add noise
                point = base_point + rng.normal(0, ood_sigma, size=(1, 2))
                points.append(point)
            samples.append(np.vstack(points))
            
        else:
            raise ValueError(f"Unknown OOD region kind: {region.kind}")
    
    if not samples:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack(samples).astype(np.float32)


def simulate_dataset(cfg: Dict[str, Any]):
    """
    Generate a toy 2D classification dataset with aleatoric/epistemic factors.

    Expected cfg keys (all optional unless noted):
        N_train, N_test (int)
        N_ood_test (int)
        tau (float > 0)
        rcd (relative class distance = d_between / sigma_within, float)
        eta (label noise rate, float in [0, 1])
        blob_sigma (float)
        class_priors (list of floats, optional)
        sigma_in or sigma_in_train/test (float)
        biases (list of 3 floats)
        undersampling (dict with boundary_band/holes)
        ood_specs (dict with ring/squares)
        seed (int)
    """
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    n_train = int(cfg.get("N_train", 1000))
    n_test = int(cfg.get("N_test", 500))
    n_classes = int(cfg.get("num_classes", 3))

    tau = float(cfg.get("tau", 0.2))
    blob_sigma = float(cfg.get("blob_sigma", 0.25))
    rcd = float(cfg.get("rcd", 3.0))  # relative class distance
    eta = float(cfg.get("eta", 0.0))  # label noise rate

    sigma_in = float(cfg.get("sigma_in", 0.0))
    sigma_in_train = float(cfg.get("sigma_in_train", sigma_in))
    sigma_in_test = float(cfg.get("sigma_in_test", sigma_in))

    # Unit equilateral triangle centered at origin (edge length = 1)
    # Vertices: top, bottom-left, bottom-right
    base_centers = np.array([
        [0.0, 1.0 / np.sqrt(3)],           # top vertex
        [-0.5, -1.0 / (2 * np.sqrt(3))],   # bottom-left
        [0.5, -1.0 / (2 * np.sqrt(3))]     # bottom-right
    ], dtype=np.float32)
    
    # Scale centers by RCD * blob_sigma to achieve desired class separation
    # RCD = d_between / sigma_within, so d_between = RCD * sigma_within
    centers = base_centers * rcd * blob_sigma
    biases = np.asarray(cfg.get("biases", [0.0, 0.0, 0.0]), dtype=np.float32)
    class_priors = _normalize_class_priors(cfg.get("class_priors"), n_classes)

    # Sample ID data (Gaussian blobs with hard labels)
    X_train_clean, y_train = _sample_gaussian_blobs(n_train, centers, blob_sigma, class_priors, rng)
    X_test_id_clean, y_test_id = _sample_gaussian_blobs(n_test, centers, blob_sigma, class_priors, rng)

    # Boundary enrichment sampling
    boundary_cfg = cfg.get("boundary_enrichment", {})
    boundary_meta = {"enabled": False, "n": 0}
    if boundary_cfg.get("enabled", False):
        n_boundary = int(boundary_cfg.get("n", 200))
        boundary_spread = float(boundary_cfg.get("spread", 1.0))
        boundary_width = float(boundary_cfg.get("width", 0.3))
        
        X_boundary, y_boundary = _sample_boundary_points(
            n_boundary, centers, blob_sigma, boundary_spread, boundary_width, rng
        )
        
        # Append to training data
        X_train_clean = np.vstack([X_train_clean, X_boundary])
        y_train = np.concatenate([y_train, y_boundary])
        
        boundary_meta = {
            "enabled": True,
            "n": n_boundary,
            "spread": boundary_spread,
            "width": boundary_width,
        }

    # Compute probabilities and labels
    scores_train = _compute_scores(X_train_clean, centers, tau, biases)
    scores_test_id = _compute_scores(X_test_id_clean, centers, tau, biases)

    probs_train = _softmax(scores_train)
    probs_test_id = _softmax(scores_test_id)

    # Apply undersampling on clean train coordinates
    undersampling_cfg = cfg.get("undersampling", {})
    keep_mask, undersampling_meta = _apply_undersampling(X_train_clean, y_train, scores_train, undersampling_cfg, rng)
    X_train_clean = X_train_clean[keep_mask]
    y_train = y_train[keep_mask]
    probs_train = probs_train[keep_mask]

    # Apply label noise to training labels
    y_train = _apply_label_noise(y_train, eta, n_classes, rng)

    # Apply input noise
    X_train = _apply_input_noise(X_train_clean, sigma_in_train, rng)
    X_test_id = _apply_input_noise(X_test_id_clean, sigma_in_test, rng)

    # OOD test data - True OOD: assign label -1 (no class assignment)
    ood_regions = _parse_ood_regions(cfg)
    # Pass training centers to OOD regions that need them
    for region in ood_regions:
        if region.kind == "confident_ood":
            region.params["training_centers"] = centers.tolist()
    X_test_ood_clean = _sample_ood_points(ood_regions, rng)
    if len(X_test_ood_clean) > 0:
        # Compute probabilities for analysis (what model would predict), but don't assign labels
        scores_test_ood = _compute_scores(X_test_ood_clean, centers, tau, biases)
        probs_test_ood = _softmax(scores_test_ood)
        # Assign -1 label to indicate true OOD (no class assignment)
        y_test_ood = np.full(len(X_test_ood_clean), -1, dtype=np.int64)
        X_test_ood = _apply_input_noise(X_test_ood_clean, sigma_in_test, rng)
    else:
        X_test_ood = np.empty((0, 2), dtype=np.float32)
        y_test_ood = np.empty((0,), dtype=np.int64)
        probs_test_ood = None

    # Combine test sets and create OOD mask
    X_test = np.vstack([X_test_id, X_test_ood]).astype(np.float32)
    y_test = np.concatenate([y_test_id, y_test_ood]).astype(np.int64)
    ood_mask = np.concatenate(
        [np.zeros(len(X_test_id), dtype=bool), np.ones(len(X_test_ood), dtype=bool)]
    )

    meta = {
        "centers": centers,
        "biases": biases,
        "tau": tau,
        "rcd": rcd,
        "eta": eta,
        "blob_sigma": blob_sigma,
        "sigma_in_train": sigma_in_train,
        "sigma_in_test": sigma_in_test,
        "undersampling": undersampling_meta,
        "boundary_enrichment": boundary_meta,
        "ood_regions": [r.__dict__ for r in ood_regions],
        "class_probs_train": probs_train,
        "probs_test_ood": probs_test_ood,  # Store OOD probabilities for analysis
        "ood_mask_test": ood_mask,
        "counts": {
            "train": len(X_train),
            "test_id": len(X_test_id),
            "test_ood": len(X_test_ood),
            "test_total": len(X_test),
        },
    }

    return (
        X_train.astype(np.float32),
        y_train.astype(np.int64),
        X_test.astype(np.float32),
        y_test.astype(np.int64),
        meta,
    )


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    """Create a 2D rotation matrix for the given angle in degrees."""
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)


def simulate_rotation_ood_dataset(cfg: Dict[str, Any]):
    """
    Generate 2D binary Gaussian classification dataset with rotation-based OOD.
    
    This creates a transformed-distribution OOD shift where OOD data is
    created by rotating the in-distribution training samples around the origin.
    
    Config keys (matching RCD style):
        N_train: Total training samples (split equally per class, default 2000)
        num_classes: Number of classes (default 2)
        class_centers: list of [x, y] centers (default [[0, 0], [3, 3]])
        blob_sigma: standard deviation for Gaussian blobs (default sqrt(0.2) ~ 0.447)
        rotation_angle: degrees to rotate for OOD (default 45)
        seed: random seed (default 42)
    
    Returns:
        Tuple of (X_train, y_train, X_train_ood, y_train_ood, meta)
        
        - X_train: Training features (ID) [N_train, 2]
        - y_train: Training labels [N_train]
        - X_train_ood: Rotated training features (OOD) [N_train, 2]
        - y_train_ood: OOD labels [N_train] (same as y_train, class structure preserved)
        - meta: Dictionary with dataset metadata
    """
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    
    # Dataset size - total samples split equally per class
    n_train = int(cfg.get("N_train", 2000))
    
    # Class centers - default to binary classification at (0,0) and (3,3)
    default_centers = [[0.0, 0.0], [3.0, 3.0]]
    centers = np.array(cfg.get("class_centers", default_centers), dtype=np.float32)
    num_classes = int(cfg.get("num_classes", len(centers)))
    
    # Ensure centers match num_classes
    if len(centers) != num_classes:
        centers = centers[:num_classes]
    
    n_per_class = n_train // num_classes
    
    # Blob standard deviation - use blob_sigma to match RCD
    blob_sigma = float(cfg.get("blob_sigma", np.sqrt(0.2)))
    
    # Rotation angle for OOD
    rotation_angle = float(cfg.get("rotation_angle", 45.0))
    R = _rotation_matrix(rotation_angle)
    
    # Sample training data - equal samples per class
    X_train_list = []
    y_train_list = []
    for class_idx in range(num_classes):
        X_class = rng.normal(
            loc=centers[class_idx],
            scale=blob_sigma,
            size=(n_per_class, 2)
        )
        y_class = np.full(n_per_class, class_idx, dtype=np.int64)
        X_train_list.append(X_class)
        y_train_list.append(y_class)
    
    X_train = np.vstack(X_train_list).astype(np.float32)
    y_train = np.concatenate(y_train_list).astype(np.int64)
    
    # Shuffle training data
    train_indices = rng.permutation(len(X_train))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    # Create OOD data by rotating training data around origin
    # X_ood = X_train @ R.T applies the rotation
    X_train_ood = (X_train @ R.T).astype(np.float32)
    y_train_ood = y_train.copy()  # Labels remain the same (same class structure)
    
    # Compute rotated centers for reference
    centers_rotated = (centers @ R.T).astype(np.float32)
    
    # Build metadata
    meta = {
        "centers": centers,
        "centers_rotated": centers_rotated,
        "blob_sigma": blob_sigma,
        "rotation_angle": rotation_angle,
        "rotation_matrix": R,
        "num_classes": num_classes,
        "counts": {
            "train": len(X_train),
            "train_per_class": n_per_class,
        },
    }
    
    return (
        X_train,
        y_train,
        X_train_ood,
        y_train_ood,
        meta,
    )


def simulate_ring_ood_dataset(cfg: Dict[str, Any]):
    """
    Generate concentric ring classification dataset with gap OOD region.
    
    Creates two rings (inner = class 0, outer = class 1) with a gap between them.
    The gap region contains no training data and is used to test epistemic uncertainty.
    
    Config keys:
        N_train: Total training samples (default 2000)
        inner_r_min: Inner ring minimum radius (default 0.5)
        inner_r_max: Inner ring maximum radius (default 1.5)
        outer_r_min: Outer ring minimum radius (default 2.5)
        outer_r_max: Outer ring maximum radius (default 3.5)
        n_gap_points: Number of gap test points (default 500)
        coord_noise_std: Std of Gaussian noise on x,y coordinates (default 0.1)
        label_noise_rate: Probability of flipping a label (default 0.05)
        seed: random seed (default 42)
    
    Returns:
        Tuple of (X_train, y_train, X_gap, meta)
        
        - X_train: Ring samples [N_train, 2]
        - y_train: Labels (0=inner, 1=outer) [N_train]
        - X_gap: Random points in gap region [n_gap_points, 2]
        - meta: Dict with radii info
    """
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    
    # Dataset size
    n_train = int(cfg.get("N_train", 2000))
    n_per_class = n_train // 2
    
    # Ring radii
    inner_r_min = float(cfg.get("inner_r_min", 0.5))
    inner_r_max = float(cfg.get("inner_r_max", 1.5))
    outer_r_min = float(cfg.get("outer_r_min", 2.5))
    outer_r_max = float(cfg.get("outer_r_max", 3.5))
    n_gap_points = int(cfg.get("n_gap_points", 500))
    
    # Noise parameters
    coord_noise_std = float(cfg.get("coord_noise_std", 0.1))
    label_noise_rate = float(cfg.get("label_noise_rate", 0.05))
    
    def sample_ring(n: int, r_min: float, r_max: float) -> np.ndarray:
        """Sample n points uniformly from a ring using polar coordinates."""
        # Sample angle uniformly [0, 2*pi]
        angles = rng.uniform(0, 2 * np.pi, size=n)
        # Sample radius uniformly in [r_min, r_max]
        # For uniform density in 2D, we should sample r^2 uniformly, then take sqrt
        # But for simplicity (and visual clarity), uniform radius sampling is fine
        radii = rng.uniform(r_min, r_max, size=n)
        # Convert to Cartesian
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1).astype(np.float32)
    
    # Sample inner ring (class 0)
    X_inner = sample_ring(n_per_class, inner_r_min, inner_r_max)
    y_inner = np.zeros(n_per_class, dtype=np.int64)
    
    # Sample outer ring (class 1)
    X_outer = sample_ring(n_per_class, outer_r_min, outer_r_max)
    y_outer = np.ones(n_per_class, dtype=np.int64)
    
    # Combine training data
    X_train = np.vstack([X_inner, X_outer])
    y_train = np.concatenate([y_inner, y_outer])
    
    # Shuffle training data
    train_indices = rng.permutation(len(X_train))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    # Add coordinate noise to training data
    if coord_noise_std > 0:
        X_train = X_train + rng.normal(0, coord_noise_std, size=X_train.shape).astype(np.float32)
    
    # Add label noise to training data
    if label_noise_rate > 0:
        flip_mask = rng.random(len(y_train)) < label_noise_rate
        y_train[flip_mask] = 1 - y_train[flip_mask]  # Flip 0->1, 1->0
    
    # Sample gap points (between inner_r_max and outer_r_min)
    gap_r_min = inner_r_max
    gap_r_max = outer_r_min
    X_gap = sample_ring(n_gap_points, gap_r_min, gap_r_max)
    
    # Build metadata
    meta = {
        "inner_r_min": inner_r_min,
        "inner_r_max": inner_r_max,
        "outer_r_min": outer_r_min,
        "outer_r_max": outer_r_max,
        "gap_r_min": gap_r_min,
        "gap_r_max": gap_r_max,
        "coord_noise_std": coord_noise_std,
        "label_noise_rate": label_noise_rate,
        "num_classes": 2,
        "counts": {
            "train": len(X_train),
            "train_per_class": n_per_class,
            "gap": n_gap_points,
        },
    }
    
    return (
        X_train,
        y_train,
        X_gap,
        meta,
    )
