# test_Plot_PDeltaX
"""画电离场的三维功率谱 P_DeltaX(k)，以及它的两个分量。

曲线含义（横轴 k 单位 Mpc^-1，纵轴 P 单位 Mpc^3）：

    P_1b(k) = x_e(1-x_e) [F(k) + G(k)]          1-bubble：气泡自身的形状/散粒噪声
    P_2b(k) = [(1-x_e)ln(1-x_e) b Ibar(k) - x_e]^2 P_m(k)
                                                2-bubble：气泡数目对暗物质密度的偏置
    P_tot   = P_1b + P_2b

其中
    F(k)    = ∫dR P(R) V(R)^2 W(kR)^2 / ∫dR P(R) V(R)
    G(k)    = ∫ d^3k'/(2π)^3 P_m(|k-k'|) F(k')
    Ibar(k) = ∫dR P(R) V(R) W(kR)   / ∫dR P(R) V(R)      (k->0 时 = 1)
    W(x)    = 3(sin x - x cos x)/x^3                     (球 top-hat)

运行：
    cd Main/Plot && python test_Plot_PDeltaX.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 让本脚本能 import 上一层 Main/bubblemodel
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bubblemodel import reionization as R
from bubblemodel import window as W
from bubblemodel import matter as Mat

# ------------------------------------------------------------------ 参数
CFG = dict(
    z=14.0,            # 红移
    z_mid=11.0,        # 再电离中点
    delta_z=7,      # 再电离宽度
    b=1.0,            # 气泡偏置
    R_bar=5.0,        # 平均气泡半径 [Mpc]
    sigma_lnR=0.5,    # lognormal 宽度
    R_min=0.02,
    R_max=40.0,
    k_min=1e-3,
    k_max=10.0,
    n_k=200,
)

SAVE = True
OUT = Path(__file__).resolve().parents[2] / "Results" / "test_plot_k3pdeltax.png"


def compute(cfg):
    """返回 (k, P_1b, P_2b, P_tot)。"""
    kc, tabs, _ = Mat.camb_linear_power([cfg["z"]])
    P_m = Mat.MatterPower(kc, tabs[cfg["z"]])

    x_e = float(R.x_e(cfg["z"], cfg["z_mid"], cfg["delta_z"]))
    k = np.logspace(np.log10(cfg["k_min"]), np.log10(cfg["k_max"]), cfg["n_k"])

    sp = W.power_spectra(
        k, x_e, P_m,
        b=cfg["b"],
        R_bar=cfg["R_bar"],
        sigma_lnR=cfg["sigma_lnR"],
        R_min=cfg["R_min"],
        R_max=cfg["R_max"],
    )
    print(f"z = {cfg['z']:g},  x_e = {x_e:.3f}")
    return sp["k"], sp["P_1b"], sp["P_2b"], sp["P_tot"]


def main():
    k, p1b, p2b, ptot = compute(CFG)
    

    plt.figure()
    plt.loglog(k, k**3*np.abs(p1b/(2*np.pi**2)), label=r"$P_{1b}$", color="blue")
    plt.loglog(k, k**3*np.abs(p2b)/(2*np.pi**2), label=r"$P_{2b}$", color="orange")
    plt.loglog(k, k**3*np.abs(ptot)/(2*np.pi**2), label=r"$P_{1b}+P_{2b}$", color="black", linestyle="--")

    plt.xlabel(r"$k$ [Mpc$^{-1}$]")
    plt.ylabel(r"$k^3P_{\Delta x}(k)/2\pi^2$ [Mpc$^3$]")
    plt.legend()

    if SAVE:
        plt.savefig(OUT, dpi=150, bbox_inches="tight")
        print(f"-> {OUT}")
    plt.show()


if __name__ == "__main__":
    main()
