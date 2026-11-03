"""第 3 层：窗函数平均 F(k)、Ī(k)，以及卷积项 G(k)。

按项目里的定义（k 为共动波数，单位 Mpc^-1）：

    F(k)  = ∫ dR P(R) [V(R)]^2 [W(kR)]^2  /  ∫ dR P(R) V(R)      [Mpc^3]
    Ī(k)  = ∫ dR P(R) V(R) W(kR)          /  ∫ dR P(R) V(R)      [无量纲]
    G(k)  = ∫ d^3k'/(2π)^3 P_m(|k - k'|) F(k')                    [Mpc^3]

注意
----
* 积分里的 P(R) 是**截断并归一化**之后的分布（见 bubbles.py），
  分母 ∫P V 就是 ⟨V_b⟩，所以 F、G、P_1b 的量纲都是 Mpc^3。
* Ī(k) 就是 2-bubble 式子里那个体积加权的窗函数平均，k→0 时 → 1。
* G 是一个**卷积**（模耦合），每个 k 都要把整条 F(k') 乘上 P_m 积一遍，
  所以它是 P_1b 里最贵的一项。下面的实现把角度积分换元成 q 积分，
  并用「预累积积分」把每个 (k, k') 的角度积分降到 O(1)：

      ∫_{-1}^{1} dμ P(|k-k'|) = (1/(k k')) ∫_{|k-k'|}^{k+k'} q P(q) dq
                              = [Q(k+k') - Q(|k-k'|)] / (k k')

  其中 Q(q) = ∫_0^q q' P(q') dq' 只需算一次。
"""

from __future__ import annotations

import numpy as np

from . import bubbles
from .maps import top_hat_window

DEFAULT_N_R = 16000          # 半径积分网格点数（对数）
DEFAULT_N_Q = 20000          # Q(q) 累积积分网格点数


# --------------------------------------------------------------------------- #
# F(k) 与 Ī(k)
# --------------------------------------------------------------------------- #
def window_moments(k, R_bar=bubbles.DEFAULT_R_BAR,
                   sigma_lnR=bubbles.DEFAULT_SIGMA_LNR,
                   R_min=bubbles.DEFAULT_R_MIN,
                   R_max=bubbles.DEFAULT_R_MAX,
                   n_R=DEFAULT_N_R):
    """一次算出 F(k) 和 Ī(k)。

    向量化方式：先把 (nk, nR) 的 W(kR) 张量算出来，再沿 R 方向梯形积分。
    内存约 nk*nR*8 字节，150 个 k × 16000 个 R 约 19 MB。
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    R = np.logspace(np.log10(R_min), np.log10(R_max), int(n_R))

    p = bubbles.lognormal_pdf(R, R_bar, sigma_lnR)
    V = bubbles.volume(R)
    norm = bubbles.pdf_norm(R_bar, sigma_lnR, R_min, R_max)
    V1 = bubbles.mean_volume(R_bar, sigma_lnR, R_min, R_max)

    W = top_hat_window(k[:, None] * R[None, :])          # (nk, nR)

    F = np.trapezoid(p[None, :] * V[None, :] ** 2 * W ** 2, R, axis=1)
    Ibar = np.trapezoid(p[None, :] * V[None, :] * W, R, axis=1)

    return F / (norm * V1), Ibar / (norm * V1)


def F_of_k(k, R_bar=bubbles.DEFAULT_R_BAR,
           sigma_lnR=bubbles.DEFAULT_SIGMA_LNR,
           R_min=bubbles.DEFAULT_R_MIN,
           R_max=bubbles.DEFAULT_R_MAX,
           n_R=DEFAULT_N_R):
    """F(k) = <V^2 W^2>/<V_b>  [Mpc^3]。"""
    return window_moments(k, R_bar, sigma_lnR, R_min, R_max, n_R)[0]


def Ibar_of_k(k, R_bar=bubbles.DEFAULT_R_BAR,
              sigma_lnR=bubbles.DEFAULT_SIGMA_LNR,
              R_min=bubbles.DEFAULT_R_MIN,
              R_max=bubbles.DEFAULT_R_MAX,
              n_R=DEFAULT_N_R):
    """Ibar(k) = <V W>/<V_b>  [无量纲]，k->0 时 -> 1。"""
    return window_moments(k, R_bar, sigma_lnR, R_min, R_max, n_R)[1]


# --------------------------------------------------------------------------- #
# G(k)：卷积项
# --------------------------------------------------------------------------- #
def G_of_k(k, P_m, F_kprime=None, kprime_grid=None,
           R_bar=bubbles.DEFAULT_R_BAR,
           sigma_lnR=bubbles.DEFAULT_SIGMA_LNR,
           R_min=bubbles.DEFAULT_R_MIN,
           R_max=bubbles.DEFAULT_R_MAX,
           kprime_min=1e-4, kprime_max=40.0, n_kprime=400,
           q_min=1e-6, q_max=60.0, n_q=DEFAULT_N_Q):
    """G(k) = ∫ d³k'/(2π)³ P_m(|k−k'|) F(k')。

    Parameters
    ----------
    k : array_like
        要计算的波数 [Mpc^-1]（必须 > 0）。
    P_m : callable
        物质功率谱，输入 k [Mpc^-1]，输出 [Mpc^3]。
    F_kprime, kprime_grid : 可选
        若已经算过 F 在 kprime_grid 上的值，直接传进来避免重复积分。
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if np.any(k <= 0):
        raise ValueError("G_of_k 只支持 k > 0")

    if kprime_grid is None:
        kprime_grid = np.logspace(np.log10(kprime_min), np.log10(kprime_max),
                                  int(n_kprime))
    if F_kprime is None:
        F_kprime = F_of_k(kprime_grid, R_bar, sigma_lnR, R_min, R_max)

    # 预先算好 Q(q) = ∫_0^q q' P(q') dq'
    q = np.logspace(np.log10(q_min), np.log10(q_max), int(n_q))
    Pq = np.asarray(P_m(q), dtype=float)
    dq = np.diff(q)
    dQ = 0.5 * (q[1:] * Pq[1:] + q[:-1] * Pq[:-1]) * dq
    Qcum = np.concatenate(([0.0], np.cumsum(dQ)))

    def Q(x):
        return np.interp(x, q, Qcum, left=0.0, right=Qcum[-1])

    out = np.empty_like(k)
    for i, ki in enumerate(k):
        # ∫dμ P(|k-k'|)，换元成 q 积分，避免 μ→±1 处的开方奇性
        angular = (Q(ki + kprime_grid) - Q(np.abs(ki - kprime_grid))) / (ki * kprime_grid)
        out[i] = np.trapezoid(kprime_grid ** 2 * F_kprime * angular, kprime_grid) / (2.0 * np.pi) ** 2

    return out


def G_zero_limit(P_m, kprime_grid, F_kprime):
    """k -> 0 时 G 的解析极限。

        G(k) = ∫d³k'/(2π)³ P(|k-k'|) F(k')
    令 k=0，则 |k-k'| -> k'：
        G(0) = ∫d³k'/(2π)³ P(k')F(k') = (1/2π²)∫k'² P(k')F(k') dk'

    注意前因子是 1/(2π²) 而不是 1/(2π)²，两者差一个因子 2：
        4π/(2π)³ = 1/(2π²)
    """
    return float(np.trapezoid(kprime_grid ** 2 * F_kprime * P_m(kprime_grid),
                              kprime_grid) / (2.0 * np.pi ** 2))


# --------------------------------------------------------------------------- #
# 功率谱三项
# --------------------------------------------------------------------------- #
def power_spectra(k, x_e, P_m, b=1.0,
                  R_bar=bubbles.DEFAULT_R_BAR,
                  sigma_lnR=bubbles.DEFAULT_SIGMA_LNR,
                  R_min=bubbles.DEFAULT_R_MIN,
                  R_max=bubbles.DEFAULT_R_MAX,
                  n_R=DEFAULT_N_R,
                  kprime_grid=None, F_kprime=None, n_kprime=400,
                  kprime_max=40.0):
    """按项目里的定义返回 P_1b、P_2b 与总和，以及各构件。

        2-bubble:  P_2b(k) = [(1-x_e) ln(1-x_e) b Ī(k) - x_e]^2 P_m(k)
        1-bubble:  P_1b(k) = x_e (1-x_e) [F(k) + G(k)]
        总功率谱:  P(k)    = P_1b(k) + P_2b(k)
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))

    F, Ibar = window_moments(k, R_bar, sigma_lnR, R_min, R_max, n_R)

    if kprime_grid is None:
        kprime_grid = np.logspace(np.log10(1e-4), np.log10(kprime_max), int(n_kprime))
    if F_kprime is None:
        F_kprime = F_of_k(kprime_grid, R_bar, sigma_lnR, R_min, R_max, n_R)

    G = G_of_k(k, P_m, F_kprime=F_kprime, kprime_grid=kprime_grid,
               R_bar=R_bar, sigma_lnR=sigma_lnR, R_min=R_min, R_max=R_max)

    x_e = float(x_e)
    bracket = (1.0 - x_e) * np.log1p(-x_e) * b * Ibar - x_e

    P_1b = x_e * (1.0 - x_e) * (F + G)
    P_2b = (bracket ** 2 - x_e ** 2) * np.asarray(P_m(k), dtype=float)

    return dict(k=k, F=F, Ibar=Ibar, G=G, bracket=bracket,
                P_1b=P_1b, P_2b=P_2b, P_tot=P_1b + P_2b)
