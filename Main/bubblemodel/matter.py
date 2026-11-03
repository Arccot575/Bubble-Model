"""第 5 层：物质功率谱 P_m(k, z)，来自 CAMB。

单位约定（这里最容易错，务必记住）
----------------------------------
CAMB 习惯用 h 作单位：k 的单位是 h/Mpc，P(k) 的单位是 (Mpc/h)^3。
本包统一用 Mpc：

    k [Mpc^-1] = k [h/Mpc] * h
    P [Mpc^3]  = P [(Mpc/h)^3] / h^3

为什么是除以 h^3 而不是乘？
    1 (Mpc/h) = 1/h Mpc，所以体积单位 (Mpc/h)^3 = Mpc^3 / h^3，
    同一个功率谱在这个单位下的数值 = 真实值 / (Mpc^3/h^3) = 真实值 * h^3。
    反推：真实值 = 数值 / h^3。

这条约定用 sigma_8 自检：由 P(k) 反算 sigma_8 应当得到 ~0.81（Planck）。
如果换算错了，sigma_8 会偏离 h^3 = 0.31 倍，一眼就能看出来。
"""

from __future__ import annotations

import numpy as np

from . import constants as C


# --------------------------------------------------------------------------- #
# CAMB
# --------------------------------------------------------------------------- #
def camb_linear_power(redshifts=(0.0, 8.0), kmax_h=100.0, npoints=1200,
                      ombh2=0.02237, omch2=0.12, H0=None, ns=0.9649,
                      verbose=False):
    """调用 CAMB 得到线性物质功率谱。

    Returns
    -------
    k : ndarray   [Mpc^-1]，升序
    tables : dict  {z: P(k)}，单位 Mpc^3
    h : float     约化哈勃常数
    """
    import camb
    from camb import model

    if H0 is None:
        H0 = C.H0_KM_S_MPC

    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns,
                           omk=0.0, tau=0.054)
    zs = sorted(float(z) for z in redshifts)
    pars.set_matter_power(redshifts=zs, kmax=kmax_h)
    pars.NonLinear = model.NonLinear_none

    res = camb.get_results(pars)
    kh, z_out, pk = res.get_matter_power_spectrum(
        minkh=1e-5, maxkh=kmax_h, npoints=npoints)

    h = pars.H0 / 100.0
    k = np.asarray(kh, float) * h                       # h/Mpc -> 1/Mpc
    tables = {float(z): np.asarray(pk[i], float) / h ** 3   # (Mpc/h)^3 -> Mpc^3
              for i, z in enumerate(z_out)}

    if verbose:
        print(f"  CAMB: h={h:.4f}, {len(k)} 个 k 点, "
              f"k in [{k[0]:.2e}, {k[-1]:.2e}] Mpc^-1")
    return k, tables, h


# --------------------------------------------------------------------------- #
# 插值器
# --------------------------------------------------------------------------- #
class MatterPower:
    """P_m(k) 的对数-对数插值器，k 单位 Mpc^-1，P 单位 Mpc^3。

    超出表格范围时用端点斜率外推（保持正数），而不是线性的 fill_value。
    """

    def __init__(self, k, P, name="P_m"):
        k = np.asarray(k, float)
        P = np.asarray(P, float)
        good = (k > 0) & (P > 0)
        self.logk = np.log(k[good])
        self.logP = np.log(P[good])
        self.name = name
        # 端点斜率，用于外推
        self.slope_lo = (self.logP[1] - self.logP[0]) / (self.logk[1] - self.logk[0])
        self.slope_hi = (self.logP[-1] - self.logP[-2]) / (self.logk[-1] - self.logk[-2])

    def __call__(self, k):
        k = np.asarray(k, float)
        lk = np.log(np.maximum(k, 1e-300))
        lP = np.interp(lk, self.logk, self.logP)
        lo = lk < self.logk[0]
        hi = lk > self.logk[-1]
        if np.any(lo):
            lP = np.where(lo, self.logP[0] + self.slope_lo * (lk - self.logk[0]), lP)
        if np.any(hi):
            lP = np.where(hi, self.logP[-1] + self.slope_hi * (lk - self.logk[-1]), lP)
        return np.exp(lP)

    @property
    def k_min(self):
        return float(np.exp(self.logk[0]))

    @property
    def k_max(self):
        return float(np.exp(self.logk[-1]))


# --------------------------------------------------------------------------- #
# sigma_R（用于自检 h^3 换算）
# --------------------------------------------------------------------------- #
def sigma_R(P_of_k, R=8.0, k_min=1e-5, k_max=120.0, n=50000):
    """sigma_R^2 = (1/2π²) ∫ k² P(k) W²(kR) dk，R 的单位是 **Mpc**。

    ⚠ 注意 R 的单位：sigma_8 里的 "8" 是 8 h^-1 Mpc，不是 8 Mpc！
    对角径 8 h^-1 Mpc 在 h=0.674 时等于 11.87 Mpc。
    直接用 R=8 Mpc 去积分会得到 1.02 而不是 0.79（差 29%），
    这是一个非常常见的坑，用 sigma8_of() 就不会踩。
    """
    from .maps import top_hat_window

    k = np.logspace(np.log10(k_min), np.log10(k_max), int(n))
    integrand = k ** 2 * P_of_k(k) * top_hat_window(k * R) ** 2
    return float(np.sqrt(np.trapezoid(integrand, k) / (2.0 * np.pi ** 2)))


def sigma8_of(P_of_k, h, k_min=1e-5, k_max=120.0, n=50000):
    """sigma_8：R = 8 h^-1 Mpc = 8/h Mpc 处的均方根密度涨落。"""
    return sigma_R(P_of_k, R=8.0 / float(h), k_min=k_min, k_max=k_max, n=n)
