"""第 2 层：气泡半径分布 P(R) 与体积统计。

这是 bubble model 真正开始"有气泡"的地方。

模型假设
--------
1. 共动体积 V0 内，气泡中心是强度为 n_bar_b 的泊松过程；
2. 气泡半径 R 独立地服从分布 P(R)（这里取对数正态）；
3. 气泡之间可以重叠，一个点只要落在任意一个气泡内就是"已电离"。

由此可以严格推出平均电离率（就是你笔记里的 eq:1-x_e）：

    <x> = 1 - exp(-n_bar_b * <V_b>)          ... (1)

    =>  n_bar_b = -ln(1 - x_e) / <V_b>       ... (2)

式 (2) 是"给定电离率，气泡数密度就被定死"的自洽性条件，答辩必问。

为什么全都用解析式，而不是数值积分？
------------------------------------
对数正态有一个非常好用的性质：作代换 u = ln R 之后，P(R)dR 恰好是
正态分布 N(ln R_bar, sigma^2) 的密度。于是任意 R^k 的矩都有闭式：

    <R^k> = R_bar^k * exp(k^2 sigma^2 / 2) * [Φ(b_hi) - Φ(b_lo)] / N

    b_hi = (ln R_max - ln R_bar - k sigma^2) / sigma      （b_lo 同理用 R_min）
    N    = Φ((ln R_max - ln R_bar)/sigma) - Φ((ln R_min - ln R_bar)/sigma)

而 <V_b^p> = (4π/3)^p <R^{3p}>。所以体积统计是**机器精度**的，
没有数值积分误差，也不会出现"图看起来差不多、其实差 1%"这种事。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, truncnorm

DEFAULT_R_BAR = 5.0            # 对数正态中位数 [Mpc]
DEFAULT_SIGMA_LNR = 0.5        # ln R 的标准差
DEFAULT_R_MIN = 0.02           # 截断下界 [Mpc]
DEFAULT_R_MAX = 40.0           # 截断上界 [Mpc]


# --------------------------------------------------------------------------- #
# 分布本身
# --------------------------------------------------------------------------- #
def lognormal_pdf(R, R_bar=DEFAULT_R_BAR, sigma_lnR=DEFAULT_SIGMA_LNR):
    """未归一化的对数正态概率密度 P(R)，单位 1/Mpc。

    R<=0 时返回 0（物理上没有负半径，数值上也避免 log 警告）。
    """
    R = np.asarray(R, dtype=float)
    out = np.zeros_like(R, dtype=float)
    good = R > 0
    if np.any(good):
        lnR = np.log(R[good])
        out[good] = (1.0 / (R[good] * sigma_lnR * np.sqrt(2.0 * np.pi)) *
                     np.exp(-(lnR - np.log(R_bar)) ** 2 / (2.0 * sigma_lnR ** 2)))
    return out


def volume(R):
    """球体积 4πR³/3 [Mpc^3]。"""
    return 4.0 / 3.0 * np.pi * np.asarray(R, dtype=float) ** 3


# --------------------------------------------------------------------------- #
# 截断归一化因子与矩（解析式）
# --------------------------------------------------------------------------- #
def _bounds(R_bar, sigma_lnR, R_min, R_max, shift=0.0):
    """u = ln R 空间里的截断边界，shift 用于算 <R^k> 时的均值平移 k*sigma^2。"""
    mu = np.log(R_bar)
    lo = (np.log(R_min) - mu - shift) / sigma_lnR
    hi = (np.log(R_max) - mu - shift) / sigma_lnR
    return lo, hi


def pdf_norm(R_bar=DEFAULT_R_BAR, sigma_lnR=DEFAULT_SIGMA_LNR,
             R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX):
    """截断区间上的归一化因子 ∫P(R)dR（解析，机器精度）。

    返回 1 表示截断没丢概率；"1 - pdf_norm" 就是丢了多少概率。
    """
    lo, hi = _bounds(R_bar, sigma_lnR, R_min, R_max)
    return float(norm.cdf(hi) - norm.cdf(lo))


def truncated_pdf(R, R_bar=DEFAULT_R_BAR, sigma_lnR=DEFAULT_SIGMA_LNR,
                  R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX):
    """归一化后的截断分布 P_trunc(R)，满足 ∫ P_trunc dR = 1。"""
    return lognormal_pdf(R, R_bar, sigma_lnR) / pdf_norm(R_bar, sigma_lnR, R_min, R_max)


def radius_moment(k, R_bar=DEFAULT_R_BAR, sigma_lnR=DEFAULT_SIGMA_LNR,
                  R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX):
    """<R^k> = ∫ P_trunc(R) R^k dR（解析式，k 可为任意实数）。"""
    lo, hi = _bounds(R_bar, sigma_lnR, R_min, R_max, shift=k * sigma_lnR ** 2)
    mean = np.exp(k * np.log(R_bar) + 0.5 * (k * sigma_lnR) ** 2)
    return float(mean * (norm.cdf(hi) - norm.cdf(lo)) / pdf_norm(
        R_bar, sigma_lnR, R_min, R_max))


def volume_moment(power, R_bar=DEFAULT_R_BAR, sigma_lnR=DEFAULT_SIGMA_LNR,
                  R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX):
    """<V_b^power> = (4π/3)^power <R^{3·power}>。

    power=1 -> [Mpc^3]；power=2 -> [Mpc^6]。
    量纲检查：1-bubble 项里 <V_b^2>/<V_b> 是 Mpc^3，正好是功率谱的量纲。
    """
    return (4.0 / 3.0 * np.pi) ** power * radius_moment(
        3.0 * power, R_bar, sigma_lnR, R_min, R_max)


def mean_volume(R_bar=DEFAULT_R_BAR, sigma_lnR=DEFAULT_SIGMA_LNR,
                R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX):
    """<V_b> [Mpc^3]。"""
    return volume_moment(1.0, R_bar, sigma_lnR, R_min, R_max)


def mean_volume_squared(R_bar=DEFAULT_R_BAR, sigma_lnR=DEFAULT_SIGMA_LNR,
                        R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX):
    """<V_b^2> [Mpc^6]。"""
    return volume_moment(2.0, R_bar, sigma_lnR, R_min, R_max)


def mean_volume_closed_form(R_bar=DEFAULT_R_BAR, sigma_lnR=DEFAULT_SIGMA_LNR):
    """<V_b> 的解析式（**不截断**时）：4π/3 * R_bar^3 * exp(4.5 σ²)。

    原理：E[R^3] = R_bar^3 e^{4.5σ²}（对数正态的矩）。
    """
    return 4.0 / 3.0 * np.pi * R_bar ** 3 * np.exp(4.5 * sigma_lnR ** 2)


def truncation_loss(R_bar=DEFAULT_R_BAR, sigma_lnR=DEFAULT_SIGMA_LNR,
                    R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX):
    """截断让 <V_b> 损失了多少（比例）。0.01 表示丢了 1%。"""
    return 1.0 - mean_volume(R_bar, sigma_lnR, R_min, R_max) / mean_volume_closed_form(
        R_bar, sigma_lnR)


# --------------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------------- #
def sample_radii(rng, n, R_bar=DEFAULT_R_BAR, sigma_lnR=DEFAULT_SIGMA_LNR,
                 R_min=DEFAULT_R_MIN, R_max=DEFAULT_R_MAX):
    """从「截断」对数正态中抽 n 个半径 [Mpc]。

    对 ln R 做截断正态抽样，用的分布和 mean_volume 完全一致 ——
    这是地图 x̄ 能和式 (1) 对上的前提。
    """
    if n <= 0:
        return np.empty(0)
    a = (np.log(R_min) - np.log(R_bar)) / sigma_lnR
    b = (np.log(R_max) - np.log(R_bar)) / sigma_lnR
    lnR = truncnorm.rvs(a, b, loc=np.log(R_bar), scale=sigma_lnR,
                        size=int(n), random_state=rng)
    return np.exp(lnR)


# --------------------------------------------------------------------------- #
# 从 x_e 反推气泡数密度
# --------------------------------------------------------------------------- #
def n_b_from_xe(x_e_value, mean_V):
    """由式 (1) 反解：n_bar_b = -ln(1 - x_e) / <V_b>。

    量纲检查：x_e 无量纲，<V_b> 单位 Mpc^3，所以 n_bar_b 单位 Mpc^-3 ✓
    """
    x_e_value = np.asarray(x_e_value, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_b = -np.log1p(-x_e_value) / mean_V
    return float(n_b) if n_b.ndim == 0 else n_b
