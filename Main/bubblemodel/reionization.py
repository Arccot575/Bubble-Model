"""第 1 层：再电离历史 x_e(z) 与平均光深 tau。

这一层只描述"宇宙平均意义上电离到多少"，不涉及任何气泡。

x_e(z) 的参数化
---------------
沿用你笔记里的 tanh 形式，但把"时间变量"取成 y = (1+z)^{3/2}
（它正比于宇宙膨胀的标度因子 a^{-3/2}，比直接用 z 更贴近物理）：

    x_e(z) = 0.5 * [ 1 - tanh( (y(z) - y_mid) / delta_y ) ]

两个物理参数：

    z_mid   : 再电离中点，x_e(z_mid) = 0.5
    delta_z : 再电离持续时标（以 z 为单位，x_e 从 0.9 降到 0.1 大致跨 delta_z）

与原代码的对应关系：

    y_re    = (1 + z_mid)^{3/2}
    delta_y = delta_z * 1.5 * sqrt(1 + z_mid)

例如原代码的 y_re=36.5, delta_y=19  →  z_mid ≈ 9.9, delta_z ≈ 3.8

平均光深
--------
    tau = sigma_T * n_p0 * ∫ x_e(z) (1+z)^2 c dz / H(z)

推导（三步，值得自己推一遍）：
    tau = sigma_T ∫ n_e^phys dl            光深的定义，dl 是物理长度
    dl  = a dchi,  n_e^phys = n_p0 x_e / a^3
      => tau = sigma_T * n_p0 ∫ x_e a^{-2} dchi
    dchi = c dz / H(z),  a^{-2} = (1+z)^2
      => tau = sigma_T * n_p0 ∫ x_e (1+z)^2 c dz / H(z)
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def y_of_z(z):
    """y(z) = (1+z)^{3/2}。"""
    return (1.0 + np.asarray(z, dtype=float)) ** 1.5


def y_of_zmid(z_mid):
    return (1.0 + float(z_mid)) ** 1.5


def delta_y_of_delta_z(z_mid, delta_z):
    """把"以 z 为单位的时标"换算成"以 y 为单位的时标"。"""
    return float(delta_z) * 1.5 * np.sqrt(1.0 + float(z_mid))


def x_e(z, z_mid=8.0, delta_z=2.0):
    """平均电离率 x_e(z)，标量或数组都可以。

    Parameters
    ----------
    z : float or array_like
        红移。
    z_mid : float
        再电离中点（x_e = 0.5 处的红移）。
    delta_z : float
        再电离持续时标（以 z 为单位）。delta_z 越大，再电离越"慢"。

    Returns
    -------
    float or ndarray
        电离率，范围 [0, 1]。
    """
    z = np.asarray(z, dtype=float)
    y = y_of_z(z)
    dy = delta_y_of_delta_z(z_mid, delta_z)
    return 0.5 * (1.0 - np.tanh((y - y_of_zmid(z_mid)) / dy))


def z_range_of_reionization(z_mid=8.0, delta_z=2.0, level=(0.1, 0.9)):
    """求 x_e 达到指定水平的红移，返回 (z_high, z_low)，用二分法。

    用于回答"这次再电离发生在 z 多少到多少之间"。
    """
    z_hi, z_lo = level

    def solve(target):
        lo, hi = 0.0, 60.0
        # x_e 随 z 单调递减，用二分
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if x_e(mid, z_mid, delta_z) > target:
                lo = mid      # 还没电离够 -> 需要更低红移
            else:
                hi = mid
        return 0.5 * (lo + hi)

    return solve(z_hi), solve(z_lo)


def optical_depth(z_mid=8.0, delta_z=2.0, z_max=80.0, n_z=8000,
                  n_e_over_n_p=C.N_E_OVER_N_P_DEFAULT):
    """平均光深 tau。

    数值上就是一条一维积分，但要保证 z_max 足够大（z_max=80 时
    还没电离的高红移部分贡献可以忽略，可自己把 z_max 加到 200 验证收敛）。
    """
    z = np.linspace(0.0, z_max, n_z)
    xe = x_e(z, z_mid, delta_z)
    # integrand 单位是 Mpc： (1+z)^2 * c/H(z)
    integrand = xe * (1.0 + z) ** 2 * C.C_KM_S / C.hubble(z)
    tau = C.SIGMA_T * C.N_P0 * n_e_over_n_p * np.trapezoid(integrand, z)
    return float(tau)


def summary(z_mid=8.0, delta_z=2.0):
    z_hi, z_lo = z_range_of_reionization(z_mid, delta_z)
    tau = optical_depth(z_mid, delta_z)
    print(f"  z_mid={z_mid:.2f}  delta_z={delta_z:.2f}")
    print(f"    x_e: 0.9 -> 0.5 -> 0.1 发生在 z = {z_hi:.2f} -> {z_mid:.2f} -> {z_lo:.2f}")
    print(f"    tau = {tau:.4f}   (Planck 2018: 0.054 +/- 0.007)")
    return tau
