import numpy as np
from typing import Tuple
from scipy.spatial import cKDTree


def compute_ldcv(points: np.ndarray, density_radius: float = 0.15) -> Tuple[float, float, float]:
    """
    计算局部密度变异系数（Local Density Coefficient of Variation, LDCV）。

    定义：
        对给定区域 R 内的点集 points，令每个点 p_i 的局部密度 d_i 为其在半径 density_radius
        邻域内（包含自身）的点数量。则：
            μ_d = (1/|R|) * Σ d_i
            σ_d = sqrt( (1/|R|) * Σ (d_i - μ_d)^2 )
            c_l = σ_d / μ_d
        当 μ_d≈0 时返回 0 以避免除零。

    参数:
        points: 区域 R 内的点集，形状为 (N, 3)。
        density_radius: 计算局部密度的邻域半径。

    返回:
        (ldcv, mu_d, sigma_d)
    """
    if points is None or len(points) == 0:
        return 0.0, 0.0, 0.0

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是形状为 (N, 3) 的数组")

    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=density_radius, return_length=True)
    counts = np.asarray(counts, dtype=np.float32)

    mu_d = float(np.mean(counts))
    sigma_d = float(np.std(counts))

    if mu_d <= 1e-12:
        ldcv = 0.0
    else:
        ldcv = float(sigma_d / mu_d)

    return ldcv, mu_d, sigma_d