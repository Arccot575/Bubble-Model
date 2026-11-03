"""画出 1-bubble、2-bubble 与总功率谱。

模型（k 为共动波数，单位 Mpc^-1）：

    P_2b(k) = [ (1-x_e) ln(1-x_e) b * Ī(k) - x_e ]^2 * P_m(k)
    F(k)    = ∫dR P(R) V(R)^2 W(kR)^2 / ∫dR P(R) V(R)
    G(k)    = ∫ d^3k'/(2π)^3 P_m(|k-k'|) F(k')
    P_1b(k) = x_e (1-x_e) [F(k) + G(k)]
    P_tot   = P_1b + P_2b

其中 Ī(k) = ∫dR P(R) V(R) W(kR) / ∫dR P(R) V(R)   （k→0 时 → 1）

产出
----
Results/fig6_power_spectra.png
    (a) P_1b、P_2b、P_tot 三条曲线
    (b) 构件 F(k) 与 G(k)
    (c) Ī(k) 与 2-bubble 的前因子 [ (1-x_e)ln(1-x_e)bĪ - x_e ]

Results/fig7_1b_2b_crossover.png
    σ_lnR = 0.2 / 0.3 / 0.5 三组 (P_1b, P_2b)，标出交叉点

跑法（在 Main/ 目录下）
-----------------------
    /home/arc/miniforge3/envs/COSMIC/bin/python run_power_spectra.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bubblemodel import bubbles as B
from bubblemodel import constants as C
from bubblemodel import matter as Mat
from bubblemodel import reionization as R
from bubblemodel import window as W

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================== #
CONFIG = dict(
    z=8.0,              # 画图所用的红移
    z_mid=8.0,          # 再电离中点（决定 x_e(z)）
    delta_z=2.0,        # 再电离时标
    b=1.0,              # 气泡偏置
    R_bar=5.0,          # 气泡半径中位数 [Mpc]
    sigma_lnR=0.5,      # ln R 宽度
    R_min=0.02,
    R_max=40.0,
    k_min=1e-5,         # 画图 k 范围 [Mpc^-1]
    k_max=10.0,
    n_k=180,
)


# =========================================================================== #
def validate(cfg, P_z0, P_z, spectra, h):
    """新一层的验收。和 selftest 同样的规矩：不过就别用这个结果。"""
    print("\n--- 第 3/5 层验收 ---")
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))

    s8 = Mat.sigma8_of(P_z0, h)
    check("sigma_8(z=0) ≈ 0.79（同时验证 h 单位换算与 R=8h^-1 Mpc）",
          abs(s8 - 0.79) < 0.03, f"sigma_8 = {s8:.4f}, CAMB 自报 0.7917")

    V1 = B.mean_volume(cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])
    V2 = B.mean_volume_squared(cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])
    F0 = float(W.F_of_k(np.array([1e-4]), cfg["R_bar"], cfg["sigma_lnR"],
                            cfg["R_min"], cfg["R_max"])[0])
    check("F(k->0) = <V^2>/<V>", abs(F0 / (V2 / V1) - 1.0) < 1e-3,
          f"F0={F0:.4e}, <V^2>/<V>={V2 / V1:.4e} Mpc^3")

    I0 = float(W.Ibar_of_k(np.array([1e-4]), cfg["R_bar"], cfg["sigma_lnR"],
                               cfg["R_min"], cfg["R_max"])[0])
    check("Ibar(k->0) = 1", abs(I0 - 1.0) < 1e-3, f"Ibar0 = {I0:.6f}")

    # G 的两条独立路径：卷积实现 vs k->0 解析极限
    kp = np.logspace(-4, np.log10(40.0), 400)
    Fkp = W.F_of_k(kp, cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])
    G_small = float(W.G_of_k(np.array([1e-4]), P_z, F_kprime=Fkp,
                             kprime_grid=kp)[0])
    G_lim = W.G_zero_limit(P_z, kp, Fkp)
    check("G(k->0) 与解析极限一致（1% 内）", abs(G_small / G_lim - 1.0) < 0.01,
          f"{G_small:.4e} vs {G_lim:.4e} Mpc^3")

    check("P_1b 与 P_2b 处处为正",
          bool(np.all(spectra["P_1b"] > 0) and np.all(spectra["P_2b"] > 0)))

    # 高 k 斜率：F ∝ k^-4
    kk = np.array([5.0, 10.0])
    Fk = W.F_of_k(kk, cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])
    slope = np.log(Fk[1] / Fk[0]) / np.log(kk[1] / kk[0])
    check("F(k) 高 k 斜率 ≈ -4（球 top-hat 的特征）", abs(slope + 4.0) < 0.15,
          f"d lnF/d lnk = {slope:.3f}")

    return all(ok)


# =========================================================================== #
def main():
    cfg = CONFIG
    t0 = time.time()

    print("=" * 78)
    print("1-bubble / 2-bubble 功率谱")
    print("=" * 78)

    x_e = float(R.x_e(cfg["z"], cfg["z_mid"], cfg["delta_z"]))
    print(f"\nz = {cfg['z']:g},  x_e = {x_e:.4f},  b = {cfg['b']:g}")
    print(f"气泡参数: R_bar={cfg['R_bar']:g} Mpc, sigma_lnR={cfg['sigma_lnR']:g}, "
          f"R in [{cfg['R_min']:g}, {cfg['R_max']:g}] Mpc")

    # ------------------------------------------------------------------ CAMB
    print("\n[第 5 层] 取 CAMB 线性物质功率谱 ...")
    k_c, tabs, h = Mat.camb_linear_power([0.0, cfg["z"]], verbose=True)
    P_z0 = Mat.MatterPower(k_c, tabs[0.0], "P_m(z=0)")
    P_z = Mat.MatterPower(k_c, tabs[cfg["z"]], f"P_m(z={cfg['z']:g})")
    print(f"  sigma_8(z=0) = {Mat.sigma8_of(P_z0, h):.4f}   "
          f"(CAMB 自报 0.7917)")
    print(f"  sigma_8(z={cfg['z']:g}) = {Mat.sigma8_of(P_z, h):.4f}")

    # ------------------------------------------------------------------ 计算
    print("\n[第 3 层] 算 F(k)、Ibar(k)、G(k) 与三项功率谱 ...")
    k = np.logspace(np.log10(cfg["k_min"]), np.log10(cfg["k_max"]), cfg["n_k"])
    t1 = time.time()
    sp = W.power_spectra(k, x_e, P_z, b=cfg["b"],
                         R_bar=cfg["R_bar"], sigma_lnR=cfg["sigma_lnR"],
                         R_min=cfg["R_min"], R_max=cfg["R_max"])
    print(f"  耗时 {time.time() - t1:.1f}s")

    validate(cfg, P_z0, P_z, sp, h)

    # ------------------------------------------------------------------ 表格
    print("\n  k [Mpc^-1]      F(k)         G(k)       Ibar     P_1b        P_2b       P_tot")
    for kk in (1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0):
        i = int(np.argmin(np.abs(k - kk)))
        print(f"  {k[i]:10.3e} {sp['F'][i]:11.3e} {sp['G'][i]:11.3e} "
              f"{sp['Ibar'][i]:8.4f} {sp['P_1b'][i]:11.3e} {sp['P_2b'][i]:11.3e} "
              f"{sp['P_tot'][i]:11.3e}")

    # 交叉点
    ratio = sp["P_2b"] / sp["P_1b"]
    crossings = []
    for i in range(len(k) - 1):
        if (ratio[i] - 1.0) * (ratio[i + 1] - 1.0) < 0:
            crossings.append(float(np.exp(np.interp(
                0.0, [ratio[i], ratio[i + 1]], np.log([k[i], k[i + 1]])))))
    cross = crossings[0] if crossings else None
    if crossings:
        txt = ", ".join(f"{c:.3e}" for c in crossings)
        print(f"\n  P_1b = P_2b 的交叉尺度: k = {txt} Mpc^-1")
        for c in crossings:
            print(f"      k={c:.3e}  ->  {2 * np.pi / c:8.0f} Mpc  "
                  f"({'P_2b 更大' if np.interp(np.log(c), np.log(k), ratio) else ''})")
    else:
        print("\n  P_1b 在整个 k 范围都高于 P_2b（sigma_lnR 太大，见 fig7）")

    # ------------------------------------------------------------------ 作图
    fig, axes = plt.subplots(1, 4, figsize=(21.5, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.loglog(k, sp["P_1b"], lw=2, color="C0", label=r"$P_{1b}$ (1-bubble)")
    ax.loglog(k, sp["P_2b"], lw=2, color="C1", label=r"$P_{2b}$ (2-bubble)")
    ax.loglog(k, sp["P_tot"], lw=2.4, color="k", ls="--", label=r"$P_{1b}+P_{2b}$")
    if cross:
        ax.axvline(cross, color="gray", ls=":", lw=1)
        ax.annotate(f"$k\\approx${cross:.3f}", xy=(cross, ax.get_ylim()[0] * 5),
                    fontsize=8, color="0.35", rotation=90, va="bottom", ha="right")
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P_{\Delta x}(k)$ [Mpc$^3$]")
    ax.set_title(f"ionization power spectrum,  $z$={cfg['z']:g},  "
                 rf"$\bar x_e$={x_e:.2f}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.loglog(k, sp["F"], lw=1.8, color="C2", label=r"$F(k)$")
    ax.loglog(k, sp["G"], lw=1.8, color="C4", label=r"$G(k)$")
    ax.loglog(k, sp["F"] + sp["G"], lw=1.2, color="0.4", ls=":", label=r"$F+G$")
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"[Mpc$^3$]")
    ax.set_title(r"building blocks: $F$ and $G$", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.semilogx(k, sp["Ibar"], lw=1.8, color="C2", label=r"$\bar I(k)$")
    ax.semilogx(k, -sp["bracket"], lw=1.8, color="C3",
                label=r"$-(1-x_e)\ln(1-x_e)b\bar I + x_e$")
    ax.axhline(x_e, color="C3", ls=":", lw=1)
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel("dimensionless")
    ax.set_title(r"$\bar I$ and the 2-bubble prefactor", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.loglog(k, ratio, lw=2.0, color="C5")
    ax.axhline(1.0, color="0.45", ls="--", lw=1.2)
    for kc in crossings:
        ax.axvline(kc, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P_{2b}(k)\,/\,P_{1b}(k)$")
    ax.set_title("which term wins", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(f"bubble model power spectra   "
                 rf"($\bar R$={cfg['R_bar']:g} Mpc, $\sigma_{{\ln R}}$={cfg['sigma_lnR']:g}, "
                 rf"$b$={cfg['b']:g}, $R_{{\max}}$={cfg['R_max']:g} Mpc)",
                 fontsize=11)

    out = RESULTS_DIR / "fig6_power_spectra.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  -> {out}")
    make_fig7_crossover(cfg, P_z, x_e)
    print(f"  总耗时 {time.time() - t0:.1f}s")


def make_fig7_crossover(cfg, P_z, x_e):
    """sigma_lnR 如何决定 P_1b / P_2b 的交叉。

    P_1b 的低 k 平台 = x_e(1-x_e) <V^2>/<V>，
    而 <V^2>/<V> = (4π/3) R_bar^3 exp(13.5 sigma^2) 对 sigma 极其敏感：
    sigma 从 0.2 变到 0.5，它涨 15 倍，于是 1-bubble 项从"低 k 次要"变成"全域主导"。
    """
    print("\n[fig7] sigma_lnR 对交叉尺度的影响 ...")
    k = np.logspace(np.log10(cfg["k_min"]), np.log10(cfg["k_max"]), cfg["n_k"])

    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
    print(f"  {'sigma_lnR':>10} {'<V^2>/<V>':>12} {'P1b(低k)':>10} {'P2b(低k)':>10} {'交叉 k':>9}")

    for j, sig in enumerate((0.2, 0.3, 0.5)):
        sp = W.power_spectra(k, x_e, P_z, b=cfg["b"], R_bar=cfg["R_bar"],
                             sigma_lnR=sig, R_min=cfg["R_min"], R_max=cfg["R_max"])
        V1 = B.mean_volume(cfg["R_bar"], sig, cfg["R_min"], cfg["R_max"])
        V2 = B.mean_volume_squared(cfg["R_bar"], sig, cfg["R_min"], cfg["R_max"])

        # P_2b - P_1b 的所有变号点（可能有 0 个、1 个或 2 个）
        r = np.log(sp["P_2b"] / sp["P_1b"])
        idx = np.nonzero(np.sign(r[:-1]) != np.sign(r[1:]))[0]
        crosses = [float(np.exp(np.interp(0.0, [r[i], r[i + 1]],
                                          np.log([k[i], k[i + 1]]))))
                   for i in idx]
        if not crosses:
            desc = "P_1b 全域主导"
        elif len(crosses) == 1:
            desc = f"交叉 {crosses[0]:.3f}"
        else:
            desc = f"P_2b 主导于 {crosses[0]:.3f}-{crosses[-1]:.3f}"
        print(f"  {sig:10.2f} {V2 / V1:12.3e} {sp['P_1b'][0]:10.3e} "
              f"{sp['P_2b'][0]:10.3e}   {desc}")

        c = f"C{j}"
        ax.loglog(k, sp["P_1b"], color=c, lw=2,
                  label=rf"$P_{{1b}}$, $\sigma_{{\ln R}}$={sig:g}")
        ax.loglog(k, sp["P_2b"], color=c, lw=2, ls="--",
                  label=rf"$P_{{2b}}$, $\sigma_{{\ln R}}$={sig:g}")
        for cc in crosses:
            y = np.exp(np.interp(np.log(cc), np.log(k), np.log(sp["P_1b"])))
            ax.plot([cc], [y], "o", color=c, ms=7, mec="k", zorder=5)

    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P_{\Delta x}(k)$ [Mpc$^3$]")
    ax.set_title(rf"$P_{{1b}}$ (solid) vs $P_{{2b}}$ (dashed), $\bar R$={cfg['R_bar']:g} Mpc, "
                 rf"$b$={cfg['b']:g}, $z$={cfg['z']:g}", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3, which="both")

    out = RESULTS_DIR / "fig7_1b_2b_crossover.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")
    print("  圆点 = P_1b 与 P_2b 的交叉点。sigma_lnR 越大，低 k 的 1-bubble "
          "平台越高，2-bubble 的主导区间被挤没。")


if __name__ == "__main__":
    main()
