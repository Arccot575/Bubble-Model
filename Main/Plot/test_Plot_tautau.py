# test_Plot_tautau
"""画 patchy reionization 的光学深度角功率谱 C_l^{tau tau}，以及 1b / 2b 分量。

用 Limber 近似把三维电离场功率谱投影到天球：

    C_l^{tau tau} = ∫ dchi  W_tau(chi)^2 / chi^2  * P_DeltaX(k = l/chi, z(chi))
    W_tau(chi)    = sigma_T * n_e0 * (1+z)^2 * x_bar_e(z)      [1/Mpc]

纵轴画的是 l(l+1)C_l/2pi（无量纲）。
1-bubble 项是天球上的白噪声（气泡散粒噪声），2-bubble 项来自大尺度偏置。

注意：Limber 在 l <~ 10 不可靠，那一段不要当真。

运行：
    cd Main/Plot && python test_Plot_tautau.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bubblemodel import constants as C
from bubblemodel import reionization as R
from bubblemodel import window as W
from bubblemodel import matter as Mat

# ------------------------------------------------------------------ 参数
CFG = dict(
    z_mid=8.0,        # 再电离中点
    delta_z=2.0,      # 再电离宽度
    b=1.0,
    R_bar=5.0,
    sigma_lnR=0.5,
    R_min=0.02,
    R_max=40.0,
    z_lo=5.0,         # 积分区间（沿视线）
    z_hi=16.0,
    n_z=45,
    ell_min=2.0,
    ell_max=2000.0,
    n_ell=60,
)

SAVE = True
OUT = Path(__file__).resolve().parents[2] / "Results" / "test_plot_tautau.png"


def compute(cfg):
    """返回 (ell, C_1b, C_2b, C_tot)。"""
    # 沿视线的红移切片
    zg = np.linspace(cfg["z_lo"], cfg["z_hi"], cfg["n_z"])
    kc, tabs, _ = Mat.camb_linear_power(list(zg))

    chi = np.array([C.comoving_distance(z) for z in zg])
    x_e = np.array([R.x_e(z, cfg["z_mid"], cfg["delta_z"]) for z in zg])
    W_tau = C.SIGMA_T * C.N_P0 * (1.0 + zg) ** 2 * x_e      # [1/Mpc]

    # 窗函数与 P(R) 的矩只在 k 上算一次（与 z 无关）
    kgrid = np.logspace(-4.5, 0.0, 200)
    F = W.F_of_k(kgrid, cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])
    Ibar = W.Ibar_of_k(kgrid, cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])

    P_m = {z: Mat.MatterPower(kc, tabs[z]) for z in zg}

    # G(k,z) = ∫ d³k'/(2π)³ P_m(k',z) F(k') 依赖 P_m，所以每个 z 单独积一次
    kp = np.logspace(-4.0, np.log10(40.0), 400)
    F_kp = W.F_of_k(kp, cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])
    G = {z: W.G_of_k(kgrid, P_m[z], F_kprime=F_kp, kprime_grid=kp) for z in zg}

    ell = np.unique(np.round(np.logspace(np.log10(cfg["ell_min"]),
                                         np.log10(cfg["ell_max"]), cfg["n_ell"])))
    c1 = np.zeros_like(ell, dtype=float)
    c2 = np.zeros_like(ell, dtype=float)

    for j, L in enumerate(ell):
        k = L / chi
        # 1-bubble: x_e(1-x_e) [F(k) + G(k)]
        f_k = np.interp(np.log(k), np.log(kgrid), F)
        g_k = np.array([np.interp(np.log(kk), np.log(kgrid), G[z])
                        for z, kk in zip(zg, k)])
        p1 = x_e * (1.0 - x_e) * (f_k + g_k)
        # 2-bubble: [ (1-x_e)ln(1-x_e) b Ibar - x_e ]^2 P_m(k,z)
        ibar = np.interp(np.log(k), np.log(kgrid), Ibar)
        pref = (1.0 - x_e) * np.log(1.0 - x_e) * cfg["b"] * ibar - x_e
        p2 = pref ** 2 * np.array([P_m[z](kk) for z, kk in zip(zg, k)])

        w2 = W_tau ** 2 / chi ** 2
        c1[j] = np.trapezoid(w2 * p1, chi)
        c2[j] = np.trapezoid(w2 * p2, chi)

    print(f"tau_bar = {float(R.optical_depth(cfg['z_mid'], cfg['delta_z'])):.4f},  "
          f"{len(ell)} 个 ell")
    return ell, c1, c2, c1 + c2


def main():
    ell, c1, c2, ctot = compute(CFG)
    y = 2 * np.pi

    plt.figure()
    plt.loglog(ell, ell * (ell + 1) * c1 / y, label=r"$1$-bubble", color="blue")
    plt.loglog(ell, ell * (ell + 1) * c2 / y, label=r"$2$-bubble", color="orange")
    plt.loglog(ell, ell * (ell + 1) * ctot / y, label=r"total", color="black",
               linestyle="--")

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\ell(\ell+1)C_\ell^{\tau\tau}/2\pi$")
    plt.legend()

    if SAVE:
        plt.savefig(OUT, dpi=150, bbox_inches="tight")
        print(f"-> {OUT}")
    plt.show()


if __name__ == "__main__":
    main()
