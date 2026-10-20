"""bubble model 再电离模拟 —— 最小可运行演示。

做什么
------
按「第 1 层 -> 第 2 层 -> 第 4 层」的顺序，依次产出：

  fig1_xe_history.png        再电离历史 x_e(z)，以及平均光深 tau 与 Planck 的对比
  fig2_bubble_pdf.png        气泡半径分布 P(R)，以及 R^3 权重如何放大长尾
  fig3_ionization_maps.png   5 个红移的电离图切片 —— 这就是"再电离过程"
  fig4_volume_fraction.png   图上实测的 x̄ 与解析式 1-exp(-n_b<V_b>) 的对比
  fig5_power_spectrum.png    图上测出的 P(k) 与布尔模型解析式的对比

不做什么
--------
不涉及窗函数平均、CAMB、1-bubble/2-bubble、C_l。那些是第 3/5/6/7 层，
等这一层的地图能稳定复现解析 x̄ 之后再做（否则解析式错了没人给你判卷）。

跑法（在 Main/ 目录下）
-----------------------
    /home/arc/miniforge3/envs/COSMIC/bin/python run_reionization_demo.py

改参数：只改下面 CONFIG 这一处。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")           # WSL 无显示器，必须用非交互后端
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bubblemodel import bubbles as B
from bubblemodel import constants as C
from bubblemodel import maps as M
from bubblemodel import reionization as R

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================== #
# 所有可调参数集中在这里（不要再写模块级全局变量！）
# =========================================================================== #
CONFIG = dict(
    # --- 气泡模型（第 2 层）---
    R_bar=5.0,          # 对数正态中位数 [Mpc]
    sigma_lnR=0.5,      # ln R 宽度；详情见 bubbles.py 顶部关于截断的警告
    R_min=0.02,         # 截断下界 [Mpc]
    R_max=40.0,         # 截断上界 [Mpc]；必须 << box_size

    # --- 再电离历史（第 1 层）---
    z_mid=8.0,          # 再电离中点
    delta_z=2.0,        # 再电离持续时标

    # --- 地图（第 4 层）---
    box_size=250.0,     # FFT 盒子边长 [Mpc]；k_fundamental = 2pi/L ≈ 0.025
    n_grid=192,         # 网格边长；分辨率 = 250/192 ≈ 1.3 Mpc
    z_show=(12.0, 10.5, 9.5, 8.5, 7.5),
    seed=2024,
)


# =========================================================================== #
def make_fig1_xe_history(cfg):
    """第 1 层：x_e(z) 与 tau 对 Planck 的对照。"""
    print("\n[fig1] 再电离历史与光深 ...")
    z = np.linspace(5.0, 20.0, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    ax = axes[0]
    for zmid, dz in [(6.0, 1.5), (8.0, 2.0), (10.0, 2.5), (12.0, 3.0)]:
        ax.plot(z, R.x_e(z, zmid, dz), label=rf"$z_{{\rm mid}}$={zmid:g}, $\Delta z$={dz:g}")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"mean ionized fraction $\bar{x}_e$")
    ax.set_title("reionization history")
    ax.set_xlim(5, 20)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    zmid_grid = np.linspace(5.0, 14.0, 46)
    taus = np.array([R.optical_depth(zm, cfg["delta_z"]) for zm in zmid_grid])
    ax.plot(zmid_grid, taus, lw=2, color="C0", label="bubble model")
    ax.axhspan(0.054 - 0.007, 0.054 + 0.007, color="C1", alpha=0.25,
               label=r"Planck 2018 $1\sigma$")
    ax.axhline(0.054, color="C1", ls="--", lw=1)
    t0 = R.optical_depth(cfg["z_mid"], cfg["delta_z"])
    ax.plot([cfg["z_mid"]], [t0], "o", color="k", zorder=5)
    ax.annotate(rf"default: $z_{{\rm mid}}$={cfg['z_mid']:g}" "\n" rf"$\tau$={t0:.3f}",
                xy=(cfg["z_mid"], t0), xytext=(6, 18), textcoords="offset points",
                fontsize=9)
    ax.set_xlabel(r"$z_{\rm mid}$")
    ax.set_ylabel(r"$\bar{\tau}$")
    ax.set_title(r"mean optical depth vs reionization midpoint")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out = RESULTS_DIR / "fig1_xe_history.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")
    return t0


# =========================================================================== #
def make_fig2_bubble_pdf(cfg):
    """第 2 层：P(R)、体积加权分布、截断损失。"""
    print("\n[fig2] 气泡半径分布与体积统计 ...")
    R_grid = np.logspace(np.log10(0.05), np.log10(60.0), 2000)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    ax = axes[0]
    for Rb in (2.0, 5.0, 10.0):
        ax.plot(R_grid, B.lognormal_pdf(R_grid, Rb, cfg["sigma_lnR"]),
                label=rf"$\bar R$ = {Rb:g} Mpc")
    ax.set_xscale("log")
    ax.set_xlabel("bubble radius $R$ [Mpc]")
    ax.set_ylabel(r"$P(R)$ [Mpc$^{-1}$]")
    ax.set_title(r"lognormal radius distribution ($\sigma_{\ln R}$="
                 f"{cfg['sigma_lnR']:g})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 体积加权分布：说明为什么 <V> 由长尾主导
    ax = axes[1]
    V1 = B.mean_volume(cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])
    w = B.lognormal_pdf(R_grid, cfg["R_bar"], cfg["sigma_lnR"]) * B.volume(R_grid) / V1
    ax.plot(R_grid, w, label=r"volume-weighted $P(R)V(R)/\langle V_b\rangle$",
            color="C3")
    ax.plot(R_grid, B.lognormal_pdf(R_grid, cfg["R_bar"], cfg["sigma_lnR"]),
            label=r"$P(R)$", color="C0", ls="--")
    ax.axvline(cfg["R_max"], color="gray", ls=":", lw=1)
    ax.text(cfg["R_max"] * 1.05, max(w) * 0.5, r"$R_{\rm max}$", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bubble radius $R$ [Mpc]")
    ax.set_ylabel("distribution")
    ax.set_title(r"$R^3$ weighting pushes $\langle V_b\rangle$ into the tail")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    loss = B.truncation_loss(cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])
    print(f"  <V_b> = {V1:.1f} Mpc^3   解析(不截断) = "
          f"{B.mean_volume_closed_form(cfg['R_bar'], cfg['sigma_lnR']):.1f} Mpc^3")
    print(f"  截断损失 = {100 * loss:.3f}%   (应当 < 1%)")

    out = RESULTS_DIR / "fig2_bubble_pdf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# =========================================================================== #
def build_maps(cfg):
    """第 4 层：为 z_show 里每个红移生成一张电离图。"""
    print("\n[fig3] 生成电离图（这一步最慢）...")
    rng = np.random.default_rng(cfg["seed"])
    data = {}
    t_start = time.time()
    for z in cfg["z_show"]:
        t0 = time.time()
        xe = float(R.x_e(z, cfg["z_mid"], cfg["delta_z"]))
        field, info = M.make_ionization_map(
            z, xe, rng, cfg["box_size"], cfg["n_grid"],
            cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])
        data[z] = (field, info)
        print(f"  z={z:5.2f}  x_e(target)={xe:7.4f}  N_b={info['n_bubbles']:6d}"
              f"  n_b<V_b>={info['occupancy']:6.3f}"
              f"  x̄(measured)={info['x_e_measured']:7.4f}"
              f"  [{time.time() - t0:4.1f}s]")
    print(f"  总耗时 {time.time() - t_start:.1f}s")
    return data


def make_fig3_ionization_maps(cfg, data):
    """第 4 层：5 个红移的切片，直观看到气泡长大、并合。"""
    from matplotlib.colors import ListedColormap

    # 注意：不要用 twilight —— 它是**循环**色图，0 和 1 是同一个颜色，
    # 二值场会被画成一片空白。用两色 ListedColormap 才可靠。
    cmap = ListedColormap(["#12122b", "#f5c451"])   # 中性=深蓝, 电离=琥珀

    n = cfg["n_grid"]
    fig, axes = plt.subplots(1, len(cfg["z_show"]),
                             figsize=(3.2 * len(cfg["z_show"]), 3.9),
                             constrained_layout=True)
    if len(cfg["z_show"]) == 1:
        axes = [axes]

    for ax, z in zip(axes, cfg["z_show"]):
        field, info = data[z]
        sl = field[n // 2].astype(float)
        im = ax.imshow(sl, origin="lower", cmap=cmap,
                       extent=[0, cfg["box_size"], 0, cfg["box_size"]],
                       interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"$z$ = {z:g}\n"
                     rf"$\bar x_e$ = {info['x_e_measured']:.3f}"
                     "\n"
                     rf"$N_b$ = {info['n_bubbles']}", fontsize=9)
        ax.set_xlabel("Mpc")
        if ax is axes[0]:
            ax.set_ylabel("Mpc")
            cb = fig.colorbar(im, ax=axes, shrink=0.85, ticks=[0.25, 0.75])
            cb.ax.set_yticklabels(["neutral", "ionized"])

    fig.suptitle("Ionization field slices  (dark = neutral, amber = ionized)", fontsize=11)
    out = RESULTS_DIR / "fig3_ionization_maps.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# =========================================================================== #
def make_fig4_volume_fraction(cfg):
    """第 4 层验收：地图实测 x̄ 必须等于 1 - exp(-n_b<V_b>)。"""
    print("\n[fig4] 检验 x̄ 与解析式 ...")
    box, n = 160.0, 96
    z_grid = np.linspace(8.0, 14.0, 6)
    n_real = 4

    V1 = B.mean_volume(cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])

    target, measured, scatter = [], [], []
    for z in z_grid:
        xe = float(R.x_e(z, cfg["z_mid"], cfg["delta_z"]))
        target.append(xe)
        vals = []
        for seed in range(n_real):
            rng = np.random.default_rng(int(z * 1000) + seed)
            field, info = M.make_ionization_map(
                z, xe, rng, box, n,
                cfg["R_bar"], cfg["sigma_lnR"], cfg["R_min"], cfg["R_max"])
            vals.append(info["x_e_measured"])
        vals = np.array(vals)
        measured.append(vals.mean())
        scatter.append(vals.std(ddof=1) / np.sqrt(len(vals)))
        print(f"  z={z:5.2f}  x_e={xe:7.4f}  measured={vals.mean():7.4f}"
              f" +/- {scatter[-1]:6.4f}")

    target = np.array(target)
    measured = np.array(measured)
    scatter = np.array(scatter)
    analytic = 1.0 - np.exp(-B.n_b_from_xe(target, V1) * V1)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True,
                             constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[3, 1]))
    ax = axes[0]
    ax.loglog(target, analytic, "k-", lw=1.5, label=r"$1-\exp(-n_b\langle V_b\rangle)$")
    ax.errorbar(target, measured, yerr=scatter, fmt="o", color="C3",
                capsize=3, label="measured on map")
    lo = min(target.min(), measured.min()) * 0.7
    hi = max(target.max(), measured.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], ls=":", color="gray", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(r"measured $\bar{x}_e$")
    ax.set_title("map vs analytic volume fraction")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.axhline(0, color="k", lw=1)
    ax.errorbar(target, measured - analytic, yerr=scatter, fmt="o", color="C3", capsize=3)
    ax.set_xlabel(r"input $x_e(z)$")
    ax.set_ylabel("residual")
    ax.grid(alpha=0.3)

    out = RESULTS_DIR / "fig4_volume_fraction.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# =========================================================================== #
def make_fig5_power_spectrum(cfg, data):
    """第 4 层验收：图上测得的 P(k) 与布尔模型解析式对比。"""
    print("\n[fig5] 功率谱：地图 vs 解析 ...")

    picks = [cfg["z_show"][0], cfg["z_show"][3]]   # 一个稀薄、一个近半电离
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

    for j, z in enumerate(picks):
        field, info = data[z]
        k, P, cnt = M.power_spectrum(field, cfg["box_size"], n_bins=20)
        color = f"C{j}"
        ax.loglog(k, P, "o-", color=color, ms=4, lw=1.2,
                  label=rf"map, $z$={z:g} ($\bar x_e$={info['x_e_measured']:.3f})")
        Pa = M.boolean_power_analytic(k, info["x_e_measured"],
                                      cfg["R_bar"], cfg["sigma_lnR"],
                                      cfg["R_min"], cfg["R_max"])
        ax.loglog(k, Pa, "--", color=color, lw=1.2, alpha=0.9,
                  label=rf"analytic (1st order), $z$={z:g}")

    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P_{\delta x}(k)$ [Mpc$^3$]")
    ax.set_title("ionization power spectrum: simulation vs analytic reference")

    # 标出"不可信区"：k 小于 5 倍基频时，盒子里容纳不下几个独立模式，
    # 而且我减掉的是本实现里的平均 x̄（k=0 模式被强行抹掉），低 k 会被系统性压低。
    k_f = 2.0 * np.pi / cfg["box_size"]
    x0, x1 = ax.get_xlim()
    ax.axvspan(x0, 5 * k_f, color="gray", alpha=0.12, zorder=0)
    ax.axvline(5 * k_f, color="gray", ls=":", lw=1)
    ax.set_xlim(x0, x1)
    ax.text(1.1 * x0, ax.get_ylim()[0] * 1.6,
            r"finite-box / undersampled" "\n" r"(do not trust below $5k_f$)",
            fontsize=7.5, color="0.35")

    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    out = RESULTS_DIR / "fig5_power_spectrum.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")
    print("  注意：解析式只在 n_b<V_b> << 1（x_e 小）时定量可靠；"
          "x_e ≈ 0.5 时它偏低，这是预期行为，不是 bug。")


# =========================================================================== #
def main():
    print("=" * 78)
    print("bubble model 再电离模拟演示")
    print("=" * 78)
    print("\n[第 0 层] 常数与单位：")
    C.summary()
    print("\n[第 1 层] 再电离历史：")
    tau = R.summary(CONFIG["z_mid"], CONFIG["delta_z"])

    t0 = time.time()
    make_fig1_xe_history(CONFIG)
    make_fig2_bubble_pdf(CONFIG)

    data = build_maps(CONFIG)
    make_fig3_ionization_maps(CONFIG, data)
    make_fig5_power_spectrum(CONFIG, data)
    make_fig4_volume_fraction(CONFIG)

    # ---------------------------------------------------------------- 汇总表
    print("\n" + "=" * 78)
    print("汇总表（建议原样贴进你的实验记录 / commit message）")
    print("=" * 78)
    V1 = B.mean_volume(CONFIG["R_bar"], CONFIG["sigma_lnR"],
                       CONFIG["R_min"], CONFIG["R_max"])
    print(f"  <V_b> = {V1:.2f} Mpc^3,  <V_b^2> = "
          f"{B.mean_volume_squared(CONFIG['R_bar'], CONFIG['sigma_lnR'], CONFIG['R_min'], CONFIG['R_max']):.3e} Mpc^6")
    print(f"  盒子 = {CONFIG['box_size']:g} Mpc, 网格 = {CONFIG['n_grid']}^3, "
          f"分辨率 = {CONFIG['box_size'] / CONFIG['n_grid']:.2f} Mpc")
    print(f"  k_fundamental = {2 * np.pi / CONFIG['box_size']:.4f} Mpc^-1")
    print(f"  tau(z_mid={CONFIG['z_mid']:g}) = {tau:.4f}")
    print()
    print(f"  {'z':>6} {'x_e(target)':>12} {'N_b':>7} {'x̄(map)':>9} {'rel.err':>9}")
    for z in CONFIG["z_show"]:
        _, info = data[z]
        rel = (info["x_e_measured"] - info["x_e_target"]) / max(info["x_e_target"], 1e-12)
        print(f"  {z:6.2f} {info['x_e_target']:12.4f} {info['n_bubbles']:7d}"
              f" {info['x_e_measured']:9.4f} {rel:9.2%}")
    print(f"\n  总耗时 {time.time() - t0:.1f}s")
    print(f"  图都存到 {RESULTS_DIR}")
    print("\n下一步：把 selftest_bubblemodel.py 跑一遍，全 PASS 之后再去动 F/G/I。")


if __name__ == "__main__":
    main()
